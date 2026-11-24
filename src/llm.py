"""Client chat unificato per Ollama e API OpenAI-compatibili (OpenRouter, OpenAI, ...)."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from logging_setup import get_logger
from providers import ProviderConfig, config_from_env, make_client

log = get_logger(__name__)


@dataclass
class CallRecord:
    time: datetime
    kind: str
    provider: str
    model: str
    latency_ms: float
    ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


# Registro globale delle chiamate API (mostrato nella pagina Log & Errori).
CALL_LOG: deque[CallRecord] = deque(maxlen=500)


def record_call(kind: str, cfg: ProviderConfig, start: float, ok: bool, prompt_tokens: int | None = None,
                completion_tokens: int | None = None, error: str | None = None) -> CallRecord:
    """Registra una chiamata API (chat o embedding) in CALL_LOG; `start` è un time.perf_counter()."""
    rec = CallRecord(time=datetime.now(), kind=kind, provider=cfg.provider, model=cfg.model,
                     latency_ms=(time.perf_counter() - start) * 1000, ok=ok,
                     prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, error=error)
    CALL_LOG.append(rec)
    return rec

# Modalità di output JSON che funziona per (provider, modello): "schema" | "json_object" | "prompt".
_JSON_MODE: dict[tuple[str, str], str] = {}
# Frasi che indicano un rifiuto del *formato* di output (non rate limit, contesto lungo, autenticazione...).
_FORMAT_HINTS = ("response_format", "json_schema", "json schema", "json_object", "structured output",
                 "structured_outputs", "strict mode", "format is not supported")


def _degrade(mode: str, error: Exception) -> str | None:
    """Prossima modalità JSON da provare, solo se l'errore è un rifiuto del formato di output:
    richiesta non valida (HTTP 400/422) il cui messaggio parla di response_format / schema JSON."""
    cause = error.__cause__ or error
    status = getattr(cause, "status_code", None)
    msg = str(cause).lower()
    if status is not None and status not in (400, 422):
        return None  # 401, 402, 429, 5xx...: il formato non c'entra
    if not any(h in msg for h in _FORMAT_HINTS):
        return None
    return {"schema": "json_object", "json_object": "prompt"}.get(mode)


def _extract_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")  # JSON dentro testo o blocchi di codice
    if start != -1 and end > start:
        try:
            out = json.loads(text[start:end + 1])
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class LLMService:
    def __init__(self, cfg: ProviderConfig | None = None, timeout: float | None = None):
        self.cfg = cfg or config_from_env("chat")
        self.model = self.cfg.model
        self.client = make_client(self.cfg, timeout, max_retries=2)

    def chat(
        self,
        messages: list[dict[str, str]],
        format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Plain chat call. If `format` is a JSON schema, structured output is requested."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("Each message must contain 'role' and 'content'")

        start = time.perf_counter()
        usage = (None, None)
        try:
            if self.cfg.is_ollama:
                kwargs: dict = {"model": self.model, "messages": messages}
                if format is not None:
                    kwargs["format"] = format
                options = {}
                if temperature is not None:
                    options["temperature"] = temperature
                if max_tokens is not None:
                    options["num_predict"] = max_tokens
                if options:
                    kwargs["options"] = options
                response = self.client.chat(**kwargs)
                usage = (getattr(response, "prompt_eval_count", None), getattr(response, "eval_count", None))
                content = response.message.content
            else:
                kwargs = {"model": self.model, "messages": messages}
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if format is not None:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "structured_response", "strict": True, "schema": format},
                    }
                elif response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    raise RuntimeError(f"Risposta senza 'choices': {response}")
                if response.usage:
                    usage = (response.usage.prompt_tokens, response.usage.completion_tokens)
                content = response.choices[0].message.content or ""

            self._record("chat", start, True, usage)
            return content
        except Exception as e:
            self._record("chat", start, False, usage, error=f"{type(e).__name__}: {e}")
            raise RuntimeError(f"{self.cfg.provider} chat failed ({self.model}): {e}") from e

    def stream_chat(self, messages: list[dict[str, str]], temperature: float | None = None):
        """Generator of text chunks (for the chatbot UI)."""
        start = time.perf_counter()
        try:
            if self.cfg.is_ollama:
                opts = {"temperature": temperature} if temperature is not None else None
                for part in self.client.chat(model=self.model, messages=messages, stream=True, options=opts):
                    yield part.message.content or ""
            else:
                kwargs = {"model": self.model, "messages": messages, "stream": True}
                if temperature is not None:
                    kwargs["temperature"] = temperature
                for chunk in self.client.chat.completions.create(**kwargs):
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            self._record("chat-stream", start, True, (None, None))
        except Exception as e:
            self._record("chat-stream", start, False, (None, None), error=f"{type(e).__name__}: {e}")
            raise RuntimeError(f"{self.cfg.provider} chat failed ({self.model}): {e}") from e

    def chat_json(self, messages: list[dict[str, str]], schema: dict, temperature: float | None = None,
                  retries: int = 1) -> dict:
        """Output strutturato con degradazione automatica:
        json_schema stretto → json_object (schema nel prompt) → testo libero con JSON estratto.
        La modalità che funziona viene ricordata per modello; se il JSON non è valido si ritenta."""
        mode = _JSON_MODE.get((self.cfg.provider, self.model), "schema")
        last_raw = ""
        for attempt in range(retries + 1):
            while True:  # scende schema -> json_object -> prompt finché una modalità è accettata
                try:
                    raw = self._chat_json_mode(messages, schema, mode, temperature)
                    break
                except RuntimeError as e:
                    next_mode = _degrade(mode, e)
                    if next_mode is None:
                        raise
                    log.info("Modello %s: modalità JSON '%s' non supportata, passo a '%s'",
                             self.model, mode, next_mode)
                    mode = _JSON_MODE[(self.cfg.provider, self.model)] = next_mode
            last_raw = raw
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
            log.warning("JSON non valido da %s (tentativo %d): %.200s", self.model, attempt + 1, raw)
        raise ValueError(f"Model did not return valid JSON. Raw response:\n{last_raw}")

    def _chat_json_mode(self, messages, schema, mode, temperature) -> str:
        if self.cfg.is_ollama or mode == "schema":
            return self.chat(messages, format=schema, temperature=temperature)
        hint = ("\n\nRespond with a single JSON object (no prose, no code fences) matching this JSON schema:\n"
                + json.dumps(schema))
        msgs = [dict(m) for m in messages]
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + hint}
        if mode == "json_object":
            return self.chat(msgs, temperature=temperature, response_format={"type": "json_object"})
        return self.chat(msgs, temperature=temperature)

    def _record(self, kind: str, start: float, ok: bool, usage, error: str | None = None) -> None:
        rec = record_call(kind, self.cfg, start, ok, usage[0], usage[1], error)
        if ok:
            log.debug("LLM %s %s ok in %.0f ms (tokens in=%s out=%s)", kind, self.model, rec.latency_ms, *usage)
        else:
            log.warning("LLM %s %s FALLITA in %.0f ms: %s", kind, self.model, rec.latency_ms, error)
