"""Calcolo degli embedding e cache su disco.

Cache separata per (dataset, modello, colonna di testo), organizzata a blocchi append-only:
    data/embeddings/<dataset_slug>/<model_slug>/ids_000001.npy   (int64)
    data/embeddings/<dataset_slug>/<model_slug>/emb_000001.npy   (float32, n x D)
    data/embeddings/<dataset_slug>/<model_slug>/meta.json
Aggiungere vettori scrive solo un nuovo blocco (niente riscrittura dell'intera cache);
`compact()` unisce i blocchi quando diventano troppi.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config import DATA_DIR, EMBEDDINGS_DIR, slugify
from llm import record_call
from logging_setup import get_logger
from providers import ProviderConfig, config_from_env, make_client

log = get_logger(__name__)

MAX_CHARS = 20_000  # tronca testi molto lunghi (≈ 5k token) per stare nei limiti dei modelli
DEFAULT_TEXT_COL = "Text"
_CHUNK_RE = re.compile(r"ids_(\d+)\.npy$")
_DIR_LOCKS: dict[str, threading.RLock] = {}
_DIR_LOCKS_GUARD = threading.Lock()


def _dir_lock(path: Path) -> threading.RLock:
    """Un lock per cartella di cache, condiviso da tutte le istanze e sessioni del processo:
    Streamlit crea un nuovo EmbeddingStore a ogni rerun, quindi un lock per istanza non proteggerebbe nulla."""
    with _DIR_LOCKS_GUARD:
        return _DIR_LOCKS.setdefault(str(path.resolve()), threading.RLock())


class EmbeddingService:
    """Generate embeddings using a local Ollama instance or an OpenAI-compatible API (thread-safe)."""

    def __init__(self, cfg: ProviderConfig | None = None, timeout: float | None = None):
        self.cfg = cfg or config_from_env("embedding")
        self.model = self.cfg.model
        self.client = make_client(self.cfg, timeout, max_retries=4)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        texts = [(t if isinstance(t, str) and t.strip() else " ")[:MAX_CHARS] for t in texts]
        start = time.perf_counter()
        tokens = None
        try:
            if self.cfg.is_ollama:
                vectors = list(self.client.embed(model=self.model, input=texts).embeddings)
            else:
                response = self.client.embeddings.create(model=self.model, input=texts)
                if not getattr(response, "data", None):
                    raise RuntimeError(f"Risposta senza dati di embedding: {response}")
                vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                if getattr(response, "usage", None):
                    tokens = response.usage.prompt_tokens
            if len(vectors) != len(texts):
                raise RuntimeError(f"Attesi {len(texts)} vettori, ricevuti {len(vectors)}")
            record_call("embedding", self.cfg, start, True, tokens)
            return vectors
        except Exception as e:
            record_call("embedding", self.cfg, start, False, tokens, error=f"{type(e).__name__}: {e}")
            raise RuntimeError(f"{self.cfg.provider} embedding failed ({self.model}): {e}") from e


def store_dirname(model: str, text_col: str = DEFAULT_TEXT_COL) -> str:
    """La colonna di testo fa parte della chiave: stessi id ma testi diversi = vettori diversi.
    La colonna di default mantiene il nome storico della cartella (compatibilità con la cache esistente)."""
    base = slugify(model)
    return base if text_col == DEFAULT_TEXT_COL else f"{base}__col-{slugify(text_col)}"


class EmbeddingStore:
    def __init__(self, dataset_slug: str, model: str, root: Path = EMBEDDINGS_DIR,
                 text_col: str = DEFAULT_TEXT_COL):
        self.dataset_slug = dataset_slug
        self.model = model
        self.text_col = text_col
        self.dir = root / slugify(dataset_slug) / store_dirname(model, text_col)
        self._lock = _dir_lock(self.dir)
        self._reset()

    def _reset(self) -> None:
        self._loaded_chunks: list[int] | None = None  # blocchi su cui è stato costruito l'indice
        self._ids: np.ndarray | None = None  # tutti gli id, nell'ordine dei blocchi
        self._chunk_of: np.ndarray | None = None  # numero di blocco per ogni id
        self._row_of: np.ndarray | None = None  # riga nel blocco
        self._index: dict[int, int] | None = None  # id -> posizione in _ids

    # ------------------------------------------------------------------ blocchi
    def _ids_path(self, chunk: int) -> Path:
        return self.dir / f"ids_{chunk:06d}.npy"

    def _emb_path(self, chunk: int) -> Path:
        return self.dir / f"emb_{chunk:06d}.npy"

    def _migrate_single_file(self) -> None:
        """Formato precedente (ids.npy/emb.npy unici) -> blocco 0."""
        old_ids, old_emb = self.dir / "ids.npy", self.dir / "emb.npy"
        if old_ids.exists() and old_emb.exists():
            old_emb.replace(self._emb_path(0))
            old_ids.replace(self._ids_path(0))

    def _chunks(self) -> list[int]:
        """Blocchi completi (ids ed emb presenti), in ordine."""
        if not self.dir.exists():
            return []
        self._migrate_single_file()
        nums = (int(m.group(1)) for p in self.dir.glob("ids_*.npy") if (m := _CHUNK_RE.search(p.name)))
        return sorted(c for c in nums if self._emb_path(c).exists())

    def _write_chunk(self, chunk: int, ids: np.ndarray, emb: np.ndarray) -> None:
        """Scrittura atomica di un blocco: prima i vettori, poi gli id (via file temporaneo + rename).
        Un blocco è considerato valido solo quando esistono entrambi: un'interruzione non lo corrompe."""
        np.save(self._emb_path(chunk), emb)
        tmp = self.dir / f"ids_{chunk:06d}.tmp.npy"
        np.save(tmp, np.asarray(ids, dtype=np.int64))
        tmp.replace(self._ids_path(chunk))

    def _load_index(self) -> dict[int, int]:
        with self._lock:
            chunks = self._chunks()
            if self._index is None or chunks != self._loaded_chunks:  # indice assente o non aggiornato
                arrays = [np.load(self._ids_path(c)) for c in chunks]
                self._loaded_chunks = chunks
                self._ids = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
                self._chunk_of = np.concatenate([np.full(len(a), c, dtype=np.int32) for c, a in zip(chunks, arrays)]) \
                    if arrays else np.empty(0, dtype=np.int32)
                self._row_of = np.concatenate([np.arange(len(a), dtype=np.int64) for a in arrays]) \
                    if arrays else np.empty(0, dtype=np.int64)
                self._index = {int(i): pos for pos, i in enumerate(self._ids)}
            return self._index

    # ------------------------------------------------------------------ lettura
    def exists(self) -> bool:
        return bool(self._chunks())

    def count(self) -> int:
        return len(self._load_index())

    def cached_ids(self) -> np.ndarray:
        self._load_index()
        return self._ids

    def dim(self) -> int | None:
        with self._lock:
            chunks = self._chunks()
            return int(np.load(self._emb_path(chunks[0]), mmap_mode="r").shape[1]) if chunks else None

    def missing_ids(self, ids) -> list[int]:
        """Id senza vettore in cache, nell'ordine dato."""
        ids = np.asarray(ids, dtype=np.int64)
        return ids[~np.isin(ids, self.cached_ids())].tolist()

    def get(self, ids) -> np.ndarray:
        """Matrix (len(ids) x D) float32 in the order of `ids`. Raises if any id is missing.

        Legge ogni blocco con indici ordinati (accesso sequenziale al file mappato in memoria:
        ~3x più veloce di un accesso casuale) e poi riordina.
        Tutto sotto lock: una compattazione concorrente non può cancellare i blocchi durante la lettura."""
        with self._lock:
            missing = self.missing_ids(ids)
            if missing:
                raise KeyError(f"{len(missing)} id senza embedding (es. {missing[:5]}) per il modello {self.model}")
            index = self._index
            pos = np.fromiter((index[int(i)] for i in ids), dtype=np.int64, count=len(ids))
            out = np.empty((len(ids), self.dim()), dtype=np.float32)
            chunk_of, row_of = self._chunk_of[pos], self._row_of[pos]
            for c in np.unique(chunk_of):
                sel = np.where(chunk_of == c)[0]
                order = np.argsort(row_of[sel])
                emb = np.load(self._emb_path(c), mmap_mode="r")
                out[sel[order]] = emb[row_of[sel][order]]
                del emb  # chiude la mappatura: su Windows un file mappato non si può cancellare
            return out

    def info(self) -> dict:
        meta_path = self.dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        size = sum(p.stat().st_size for p in self.dir.glob("*.npy")) if self.dir.exists() else 0
        return {"model": self.model, "text_col": self.text_col, "count": self.count(), "dim": self.dim(),
                "chunks": len(self._chunks()), "size_mb": round(size / 1e6, 1), "updated": meta.get("updated")}

    # ------------------------------------------------------------------ scrittura
    def append(self, ids: list[int], vectors) -> None:
        """Scrive un nuovo blocco (append-only)."""
        if not len(ids):
            return
        emb = np.asarray(vectors, dtype=np.float32)
        d = self.dim()
        if d is not None and emb.shape[1] != d:
            raise ValueError(f"Dimensione embedding incoerente: cache {d}, nuovi {emb.shape[1]}")
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._write_chunk(max(self._chunks(), default=0) + 1, ids, emb)
            self._write_meta()
            self._reset()

    def compact(self) -> None:
        """Unisce tutti i blocchi in uno solo (letture più rapide). Tutta l'operazione è sotto lock."""
        with self._lock:
            self._reset()
            chunks = self._chunks()
            if len(chunks) <= 1:
                return
            ids = self.cached_ids().copy()
            # 1) nuovo blocco completo; 2) solo dopo si rimuovono i vecchi:
            #    un'interruzione lascia duplicati identici, mai perdite
            self._write_chunk(max(chunks) + 1, ids, self.get(ids))
            for c in chunks:
                self._ids_path(c).unlink()
                self._emb_path(c).unlink()
            self._reset()
        log.info("Cache %s compattata: %d blocchi -> 1", self.model, len(chunks))

    def _write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps({
            "model": self.model, "dataset": self.dataset_slug, "text_col": self.text_col,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")

    def delete(self) -> None:
        for p in self.dir.glob("*"):
            p.unlink()
        if self.dir.exists():
            self.dir.rmdir()
        self._reset()


def list_stores(dataset_slug: str) -> list[EmbeddingStore]:
    base = EMBEDDINGS_DIR / slugify(dataset_slug)
    stores = []
    for d in sorted(base.iterdir()) if base.exists() else []:
        meta = d / "meta.json"
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        stores.append(EmbeddingStore(dataset_slug, info.get("model", d.name),
                                     text_col=info.get("text_col", DEFAULT_TEXT_COL)))
    return stores


def compute_missing(
    service: EmbeddingService,
    store: EmbeddingStore,
    df: pd.DataFrame,
    batch_size: int = 100,
    workers: int = 4,
    flush_every: int = 2000,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embeds the rows of df (columns Id, Text) not yet in the cache, with `workers` parallel requests.

    I risultati vengono salvati su disco ogni ~`flush_every` testi: in caso di errore il lavoro già
    fatto resta in cache e rilanciando si riparte da lì. Ritorna il numero di testi calcolati."""
    todo = df[df["Id"].astype(int).isin(store.missing_ids(df["Id"]))]
    total = len(todo)
    log.info("Embedding: %d testi con %s (batch %d, %d richieste parallele)", total, service.model, batch_size, workers)
    if total == 0:
        return 0

    batches = [todo.iloc[s:s + batch_size] for s in range(0, total, batch_size)]
    pending_ids: list[int] = []
    pending_vecs: list[list[float]] = []
    done = 0
    start = time.perf_counter()

    def collect(fut) -> None:
        nonlocal done
        chunk = futures[fut]
        pending_vecs.extend(fut.result())
        pending_ids.extend(chunk["Id"].astype(int).tolist())
        done += len(chunk)

    def flush() -> None:
        nonlocal pending_ids, pending_vecs
        store.append(pending_ids, pending_vecs)
        pending_ids, pending_vecs = [], []

    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="embed")
    futures: dict = {}
    first_error: BaseException | None = None
    try:
        futures = {pool.submit(service.embed_texts, b["Text"].tolist()): b for b in batches}
        remaining = set(futures)
        while remaining:
            # FIRST_COMPLETED: si aggiornano progresso e salvataggi man mano che i batch terminano
            finished, remaining = wait(remaining, return_when=FIRST_COMPLETED)
            for fut in finished:
                if fut.cancelled():
                    continue
                if fut.exception() is not None:  # si registrano comunque tutti i batch riusciti di questo giro
                    first_error = first_error or fut.exception()
                    continue
                collect(fut)
            if progress:
                progress(done, total)
            if len(pending_ids) >= flush_every:
                flush()
            if first_error is not None:
                for f in remaining:
                    f.cancel()
                # attende i batch già in volo per non buttare richieste pagate
                for fut in wait(remaining).done:
                    if not fut.cancelled() and fut.exception() is None:
                        collect(fut)
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        flush()
    if first_error is not None:
        log.warning("Embedding interrotto da un errore dopo %d/%d testi (salvati in cache)", done, total)
        raise first_error
    log.info("Embedding completato: %d testi in %.1fs", done, time.perf_counter() - start)
    if len(store._chunks()) > 20:
        store.compact()
    return done


def import_legacy_cache(dataset_slug: str, model: str, legacy_path: Path = DATA_DIR / "embeddings.npz") -> int:
    """Converts the old data/embeddings.npz (indexed only by Id) into the per-model store."""
    cache = np.load(legacy_path, allow_pickle=True)
    ids = np.asarray(cache["ids"], dtype=np.int64)
    store = EmbeddingStore(dataset_slug, model)
    mask = np.isin(ids, store.missing_ids(ids))
    store.append(ids[mask].tolist(), np.asarray(cache["embeddings"][mask], dtype=np.float32))
    log.info("Importata cache legacy: %d vettori -> %s", int(mask.sum()), store.dir)
    return int(mask.sum())


def estimate_tokens(texts: pd.Series) -> int:
    """Rough estimate (≈ 4 characters per token)."""
    return int(texts.str.len().clip(upper=MAX_CHARS).sum() / 4)
