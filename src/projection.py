"""Proiezione 2D degli embedding per la visualizzazione (PCA istantanea, UMAP più fedele).

UMAP usa numba: la prima chiamata in un processo compila il codice (~45 s). `warmup_umap()`
la esegue in un thread in background all'avvio della GUI, così quando serve è già pronta (~3 s).
"""

from __future__ import annotations

import threading

import numpy as np
from sklearn.decomposition import PCA

from logging_setup import get_logger

log = get_logger(__name__)

METHODS = {"umap": "UMAP", "pca": "PCA"}
_umap_ready = threading.Event()
_warmup_started = False
_warmup_lock = threading.Lock()


def make_umap(n_components: int, seed: int | None = None, **kwargs):
    """UMAP con le impostazioni del progetto. Con `seed` è riproducibile ma gira su un solo core;
    senza seed usa tutti i core (adatto alla proiezione 2D, solo visiva)."""
    import umap

    parallel = {"n_jobs": -1} if seed is None else {"random_state": seed, "n_jobs": 1}
    return umap.UMAP(n_components=n_components, metric="cosine", compatibility_layout=True, **parallel, **kwargs)


def _warmup() -> None:
    try:
        X = np.random.default_rng(0).normal(size=(300, 10)).astype(np.float32)
        # due percorsi di codice distinti da compilare: parallelo (proiezione 2D) e con seed (preprocessing)
        make_umap(2, n_epochs=20).fit_transform(X)
        make_umap(5, seed=0, n_epochs=20).fit_transform(X)
        log.info("UMAP pronto (compilazione completata)")
    except Exception as e:  # noqa: BLE001
        log.warning("Warm-up UMAP fallito: %s", e)
    finally:
        _umap_ready.set()


def warmup_umap() -> None:
    """Avvia (una sola volta per processo) la compilazione di UMAP in background."""
    global _warmup_started
    with _warmup_lock:
        if not _warmup_started:
            _warmup_started = True
            threading.Thread(target=_warmup, name="umap-warmup", daemon=True).start()


def umap_ready() -> bool:
    return _umap_ready.is_set()


def project_2d(X: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    log.info("Proiezione 2D %s su %s", method, X.shape)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "umap":
        return make_umap(2).fit_transform(X)  # senza seed: tutti i core (con il seed ~10x più lento)
    raise ValueError(f"Metodo di proiezione sconosciuto: {method}")
