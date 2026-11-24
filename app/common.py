"""Utility condivise dalle pagine della GUI: stato di sessione, gestione errori, cache."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import get_env  # noqa: E402
from dataset import DatasetDownloader, SampleConfig, make_sample  # noqa: E402
from embeddings import EmbeddingStore  # noqa: E402
from llm import LLMService  # noqa: E402
from logging_setup import errors_total, get_logger, record_error, setup_logging  # noqa: E402
from providers import ProviderConfig, config_from_env  # noqa: E402
from runs import RUNS_DIR, Run, load_arrays, load_run  # noqa: E402

setup_logging()
log = get_logger("gui")


# ----------------------------------------------------------------------------- stato
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("chat_cfg", config_from_env("chat"))
    ss.setdefault("emb_cfg", config_from_env("embedding"))
    ss.setdefault("sample_cfg", SampleConfig())
    ss.setdefault("sample_df", None)  # DataFrame del campione corrente
    ss.setdefault("X", None)  # embedding del campione (n x D)
    ss.setdefault("X_model", None)  # modello con cui è stato costruito X
    ss.setdefault("result", None)  # dict con il clustering corrente non ancora salvato
    ss.setdefault("chat_history", [])
    ss.setdefault("errors_seen", 0)
    ss.setdefault("llm_workers", int(get_env("LLM_CONCURRENCY", "4")))  # richieste LLM parallele
    ss.setdefault("emb_workers", int(get_env("EMBEDDING_CONCURRENCY", "4")))  # richieste embedding parallele
    ss.setdefault("label_language", get_env("LABEL_LANGUAGE", "Italian"))


LANGUAGES = {"Italian": "Italiano", "English": "English"}  # lingua dei nomi dei cluster


def language_index(current: str) -> int:
    return list(LANGUAGES).index(current) if current in LANGUAGES else 0


def set_workspace(sample_cfg: SampleConfig, sample_df: pd.DataFrame | None, X: np.ndarray | None = None,
                  X_model: str | None = None, clear_caches: bool = False) -> None:
    """Sostituisce il workspace (campione + embedding). Il risultato di clustering corrente non è più valido;
    `clear_caches` scarta anche curva di k e preprocessing in cache (necessario cambiando dataset)."""
    ss = st.session_state
    ss.sample_cfg, ss.sample_df, ss.X, ss.X_model, ss.result = sample_cfg, sample_df, X, X_model, None
    if clear_caches:
        ss.pop("k_sweep", None)
        ss.pop("prep_cache", None)


def chat_cfg() -> ProviderConfig:
    return st.session_state.chat_cfg


def emb_cfg() -> ProviderConfig:
    return st.session_state.emb_cfg


def downloader() -> DatasetDownloader:
    return DatasetDownloader()


def dataset_slug() -> str:
    return downloader().slug


def current_store(text_col: str | None = None, model: str | None = None) -> EmbeddingStore:
    """Cache per (dataset, modello, colonna di testo). Default: modello selezionato e colonna del campione."""
    col = text_col or st.session_state.sample_cfg.text_col
    return EmbeddingStore(dataset_slug(), model or emb_cfg().model, text_col=col)


@st.cache_resource(max_entries=8)
def _llm_service(provider: str, base_url: str, api_key: str, model: str) -> LLMService:
    return LLMService(ProviderConfig(provider, base_url, api_key, model))


def llm_service(cfg: ProviderConfig) -> LLMService:
    """Client LLM riusato tra i rerun (stessa connessione HTTP): evita di ricrearlo a ogni messaggio."""
    return _llm_service(cfg.provider, cfg.base_url, cfg.api_key, cfg.model)


# ----------------------------------------------------------------------------- errori
@contextmanager
def guarded(step: str, **context):
    """Esegue un blocco: in caso di eccezione la registra e la mostra in modo leggibile
    (messaggio + id errore + traceback espandibile), senza far crashare la pagina."""
    try:
        yield
    except Exception as e:  # noqa: BLE001
        entry = record_error(step, e, context)
        st.error(f"**{step}** non riuscito — `{entry.error_type}`: {entry.message}", icon="🚨")
        hint = error_hint(e)
        if hint:
            st.info(hint, icon="💡")
        with st.expander(f"Dettagli tecnici (errore {entry.id})"):
            if context:
                st.json(context)
            st.code(entry.traceback, language="python")


def error_hint(e: Exception) -> str | None:
    """Suggerimenti per gli errori più comuni."""
    msg = f"{type(e).__name__} {e}".lower()
    if "401" in msg or "unauthorized" in msg or "invalid api key" in msg or "authentication" in msg:
        return "Chiave API non valida o mancante: controlla la pagina *Modelli & API*."
    if "402" in msg or "insufficient" in msg or "credits" in msg:
        return "Credito esaurito sul provider (OpenRouter: controlla i crediti dell'account)."
    if "429" in msg or "rate limit" in msg:
        return "Limite di richieste raggiunto: riduci il batch size o riprova tra poco."
    if "404" in msg or "not found" in msg and "model" in msg:
        return "Modello non trovato sul provider: verifica il nome del modello in *Modelli & API*."
    if "timed out" in msg or "timeout" in msg:
        return ("Richiesta scaduta: il servizio è troppo lento per la quantità di testo inviata. Riduci i testi per "
                "richiesta e le richieste parallele (con Ollama locale: 1 richiesta, 8-16 testi).")
    if any(s in msg for s in ("connect", "connessione", "10061", "refused")):
        return "Problema di rete o servizio non raggiungibile (per Ollama: è avviato?)."
    if "kaggle" in msg or "kaggle_username" in msg or "kaggle_key" in msg:
        return "Credenziali Kaggle mancanti: imposta KAGGLE_USERNAME e KAGGLE_KEY nel file .env."
    if "memory" in msg:
        return "Memoria insufficiente: riduci il campione o usa la riduzione PCA."
    return None


def new_errors_count() -> int:
    return errors_total() - st.session_state.get("errors_seen", 0)


# ----------------------------------------------------------------------------- dati in cache
def _csv_mtime() -> float:
    p = downloader().csv_path
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Lettura del dataset…", max_entries=1)
def _load_raw(mtime: float) -> pd.DataFrame:
    # cache_resource: un solo DataFrame condiviso, senza copie a ogni rerun (va trattato in sola lettura)
    return downloader().load_raw()


def load_raw() -> pd.DataFrame:
    return _load_raw(_csv_mtime())


def build_sample(cfg: SampleConfig) -> pd.DataFrame:
    allowed = current_store(cfg.text_col).cached_ids() if cfg.only_cached else None
    if allowed is not None and len(allowed) == 0:
        raise ValueError(f"Nessun embedding in cache per {emb_cfg().model}: disattiva 'solo testi in cache'")
    return make_sample(load_raw(), cfg, allowed_ids=allowed)


def sample_from_ids(ids: np.ndarray, cfg_dict: dict) -> pd.DataFrame:
    """Ricostruisce il campione di un run salvato, nello stesso ordine degli id (in cache).
    Nella chiave entrano solo i campi che determinano il risultato (filtri e campionamento sono ignorati)."""
    cfg = SampleConfig.from_dict(cfg_dict).unfiltered()
    key = (cfg.text_col, cfg.id_col, tuple(cfg.extra_cols))
    return _sample_from_ids(_csv_mtime(), ids.tobytes(), key, ids, cfg)


@st.cache_data(show_spinner="Ricostruzione del campione…", max_entries=8)
def _sample_from_ids(mtime: float, ids_key: bytes, cfg_key: tuple, _ids: np.ndarray, _cfg: SampleConfig) -> pd.DataFrame:
    # solo i parametri senza '_' entrano nella chiave della cache
    ids = _ids
    full = make_sample(load_raw(), _cfg).set_index("Id")
    missing = np.setdiff1d(ids, full.index.to_numpy())
    if len(missing):
        raise KeyError(f"{len(missing)} id del run non presenti nel dataset attuale")
    return full.loc[ids].reset_index()


@st.cache_resource(max_entries=16)
def _run_arrays(run_id: str, mtime_ns: int) -> dict:
    return load_arrays(run_id)  # gli array di un set non cambiano dopo il salvataggio: condivisi in sola lettura


def load_run_cached(run_id: str) -> Run:
    """Come runs.load_run, ma senza decomprimere arrays.npz a ogni rerun (run.json resta sempre fresco)."""
    mtime_ns = (RUNS_DIR / run_id / "arrays.npz").stat().st_mtime_ns
    return load_run(run_id, arrays=_run_arrays(run_id, mtime_ns))


def workspace_ready() -> bool:
    ss = st.session_state
    return ss.sample_df is not None and ss.X is not None and len(ss.X) == len(ss.sample_df)


def load_embeddings_for_sample() -> bool:
    """Carica dalla cache gli embedding del campione corrente. True se tutti presenti."""
    ss = st.session_state
    store = current_store()
    if ss.sample_df is None or store.missing_ids(ss.sample_df["Id"]):
        return False
    ss.X = store.get(ss.sample_df["Id"].to_numpy())
    ss.X_model = store.model
    return True


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


METRIC_HELP = {
    "silhouette": "Silhouette ∈ [-1, 1]: più alto = cluster compatti e separati (calcolata su un campione).",
    "davies_bouldin": "Davies-Bouldin ≥ 0: più basso è meglio. Complementare alla silhouette.",
    "noise_frac": "Frazione di punti marcati come rumore (solo HDBSCAN).",
    "ambiguous_frac": "Frazione di punti con probabilità massima < 0.5: casi di confine, utili all'agente "
                      "per decidere cosa chiedere.",
}
