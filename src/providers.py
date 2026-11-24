"""Configurazione dei provider di modelli (chat ed embedding) ed elenco dei modelli disponibili.

Provider supportati:
- openrouter: API OpenAI-compatibile di OpenRouter (elenco modelli con prezzi);
- openai:     qualunque endpoint OpenAI-compatibile (OpenAI, vLLM, LM Studio, ...);
- ollama:     istanza Ollama locale (nessuna API key).

Variabili .env lette (le specifiche hanno priorità su quelle condivise):
    LLM_PROVIDER / LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
    EMBEDDING_PROVIDER / EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL
    BASE_URL / API_KEY  (condivise, retrocompatibili)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import requests

from config import get_env
from logging_setup import get_logger

log = get_logger(__name__)

# Proprietà di ogni provider. Ollama gira in locale (spesso su CPU) e serve una richiesta alla volta:
# timeout più ampio, nessun parallelismo, batch di embedding piccoli.
PROVIDERS = {
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "needs_key": True,
                   "timeout": 120, "serial": False, "embed_batch": 100},
    "openai": {"label": "OpenAI-compatibile", "base_url": "https://api.openai.com/v1", "needs_key": True,
               "timeout": 120, "serial": False, "embed_batch": 100},
    "ollama": {"label": "Ollama (locale)", "base_url": "http://localhost:11434", "needs_key": False,
               "timeout": 600, "serial": True, "embed_batch": 16},
}

ENV_PREFIX = {"chat": "LLM", "embedding": "EMBEDDING"}  # LLM_MODEL, EMBEDDING_BASE_URL, ...

DEFAULT_MODELS = {
    ("openrouter", "chat"): "deepseek/deepseek-chat",
    ("openrouter", "embedding"): "openai/text-embedding-3-small",
    ("openai", "chat"): "gpt-4o-mini",
    ("openai", "embedding"): "text-embedding-3-small",
    ("ollama", "chat"): "llama3.2:3b",
    ("ollama", "embedding"): "bge-m3",
}


@dataclass
class ModelInfo:
    id: str
    name: str
    context_length: int | None = None
    price_in: float | None = None  # USD per 1M token in input
    price_out: float | None = None  # USD per 1M token in output

    @property
    def label(self) -> str:
        parts = [self.id]
        if self.price_in is not None:
            price = f"${self.price_in:.3g}/M in"
            if self.price_out:
                price += f", ${self.price_out:.3g}/M out"
            parts.append(price)
        if self.context_length:
            parts.append(f"ctx {self.context_length:,}")
        return " · ".join(parts)


@dataclass
class ProviderConfig:
    provider: str
    base_url: str
    api_key: str
    model: str

    @property
    def is_ollama(self) -> bool:
        return self.provider == "ollama"

    @property
    def serial(self) -> bool:
        """True se il provider serve una richiesta alla volta (richieste parallele finirebbero solo in coda)."""
        return PROVIDERS[self.provider]["serial"]

    @property
    def default_timeout(self) -> float:
        return PROVIDERS[self.provider]["timeout"]

    @property
    def default_embed_batch(self) -> int:
        return PROVIDERS[self.provider]["embed_batch"]

    def effective_workers(self, requested: int) -> int:
        return 1 if self.serial else requested

    def with_model(self, model: str) -> "ProviderConfig":
        return replace(self, model=model)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("api_key")
        return d


def make_client(cfg: ProviderConfig, timeout: float | None = None, max_retries: int = 2):
    """Client Ollama o OpenAI-compatibile per `cfg`, con il timeout di default del provider."""
    timeout = timeout or cfg.default_timeout
    if cfg.is_ollama:
        from ollama import Client

        return Client(host=cfg.base_url, timeout=timeout)
    from openai import OpenAI

    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout, max_retries=max_retries)


def env_values(cfg: ProviderConfig, kind: str) -> dict[str, str]:
    """Variabili .env che `config_from_env` rilegge per ricostruire `cfg` (la chiave solo se presente)."""
    prefix = ENV_PREFIX[kind]
    values = {f"{prefix}_PROVIDER": cfg.provider, f"{prefix}_MODEL": cfg.model, f"{prefix}_BASE_URL": cfg.base_url}
    if cfg.api_key:
        values[f"{prefix}_API_KEY"] = cfg.api_key
    return values


def infer_provider(base_url: str, api_key: str) -> str:
    if "openrouter.ai" in base_url:
        return "openrouter"
    if not api_key or ":11434" in base_url:
        return "ollama"
    return "openai"


def config_from_env(kind: str) -> ProviderConfig:
    """kind: 'chat' or 'embedding'."""
    prefix = ENV_PREFIX[kind]
    shared_url, shared_key = get_env("BASE_URL", "http://localhost:11434"), get_env("API_KEY")
    shared_provider = infer_provider(shared_url, shared_key)
    base_url = get_env(f"{prefix}_BASE_URL") or shared_url
    own_key = get_env(f"{prefix}_API_KEY")
    # provider salvato esplicitamente dalla GUI; altrimenti dedotto (compatibilità con .env esistenti)
    provider = get_env(f"{prefix}_PROVIDER") or infer_provider(base_url, own_key or shared_key)
    if provider not in PROVIDERS:
        log.warning("%s_PROVIDER=%r non valido: lo deduco dall'URL", prefix, provider)
        provider = infer_provider(base_url, own_key or shared_key)
    # la chiave condivisa si usa solo se appartiene allo stesso provider: mai inviarla a un altro servizio
    if provider == "ollama":
        api_key = ""
    else:
        api_key = own_key or (shared_key if provider == shared_provider else "")
    model = get_env(f"{prefix}_MODEL") or DEFAULT_MODELS[(provider, kind)]
    return ProviderConfig(provider=provider, base_url=base_url, api_key=api_key, model=model)


def _price_per_million(value) -> float | None:
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def list_models(cfg: ProviderConfig, kind: str, timeout: float = 15) -> list[ModelInfo]:
    """Return the models available on the provider. Raises on network/auth errors."""
    if cfg.provider == "ollama":
        from ollama import Client

        resp = Client(host=cfg.base_url).list()
        models = [ModelInfo(id=m.model, name=m.model) for m in resp.models]
        # Ollama non distingue chat/embedding: euristica sul nome per mettere prima i modelli del tipo richiesto.
        emb_hint = ("embed", "bge", "e5", "minilm", "nomic")
        models.sort(key=lambda m: (any(h in m.id.lower() for h in emb_hint) != (kind == "embedding"), m.id))
        return models

    if cfg.provider == "openrouter":
        path = "/embeddings/models" if kind == "embedding" else "/models"
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        r = requests.get(cfg.base_url.rstrip("/") + path, headers=headers, timeout=timeout)
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            arch = m.get("architecture") or {}
            if kind == "chat" and "text" not in (arch.get("output_modalities") or ["text"]):
                continue
            pricing = m.get("pricing") or {}
            out.append(ModelInfo(
                id=m["id"],
                name=m.get("name", m["id"]),
                context_length=m.get("context_length"),
                price_in=_price_per_million(pricing.get("prompt")),
                price_out=_price_per_million(pricing.get("completion")) if kind == "chat" else None,
            ))
        out.sort(key=lambda m: m.id)
        return out

    from openai import OpenAI

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout)
    ids = sorted(m.id for m in client.models.list())
    if kind == "embedding":
        ids.sort(key=lambda i: "embed" not in i.lower())
    return [ModelInfo(id=i, name=i) for i in ids]


def test_connection(cfg: ProviderConfig, kind: str) -> tuple[bool, str]:
    """Esegue una chiamata minima reale. Ritorna (ok, messaggio)."""
    start = time.perf_counter()
    try:
        if kind == "chat":
            from llm import LLMService

            reply = LLMService(cfg).chat([{"role": "user", "content": "Reply with the single word: OK"}], max_tokens=10)
            detail = f"risposta: {reply.strip()[:40]!r}"
        else:
            from embeddings import EmbeddingService

            vec = EmbeddingService(cfg).embed_texts(["connection test"])[0]
            detail = f"dimensione vettore: {len(vec)}"
        ms = (time.perf_counter() - start) * 1000
        msg = f"OK in {ms:.0f} ms — {detail}"
        log.info("Test connessione %s (%s, %s): %s", kind, cfg.provider, cfg.model, msg)
        return True, msg
    except Exception as e:  # noqa: BLE001 - riportato alla GUI
        log.warning("Test connessione %s fallito (%s, %s): %s", kind, cfg.provider, cfg.model, e)
        return False, f"{type(e).__name__}: {e}"
