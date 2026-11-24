"""Clustering iniziale: preprocessing, algoritmi (K-Means, HDBSCAN), assegnazioni soft, metriche."""

from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_samples
from sklearn.preprocessing import normalize

from logging_setup import get_logger

log = get_logger(__name__)

ALGORITHMS = {
    "kmeans": "K-Means",
    "hdbscan": "HDBSCAN",
}


# ----------------------------------------------------------------------------- preprocessing
@dataclass
class PreprocessConfig:
    l2_normalize: bool = True
    reduction: str = "pca"  # "none" | "pca" | "umap"
    n_components: int = 50
    seed: int = 42


def preprocess(X: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if cfg.l2_normalize:
        X = normalize(X)
    n_comp = min(cfg.n_components, X.shape[1], max(1, X.shape[0] - 1))
    if cfg.reduction == "pca" and n_comp < X.shape[1]:
        X = PCA(n_components=n_comp, svd_solver="randomized", random_state=cfg.seed).fit_transform(X)
    elif cfg.reduction == "umap":
        from projection import make_umap

        if X.shape[1] > 100:  # PCA a 50 prima di UMAP: stessa struttura, ricerca dei vicini molto più rapida
            X = PCA(n_components=min(50, X.shape[0] - 1), svd_solver="randomized",
                    random_state=cfg.seed).fit_transform(X)
        # con seed: il preprocessing determina i cluster, quindi deve essere riproducibile dal seed salvato
        X = make_umap(n_comp, seed=cfg.seed).fit_transform(X)
    return np.ascontiguousarray(X, dtype=np.float32)


# ----------------------------------------------------------------------------- risultato
@dataclass
class ClusteringResult:
    algorithm: str
    params: dict
    labels: np.ndarray  # (n,) con -1 = rumore
    probs: np.ndarray | None  # (n, K) assegnazioni soft, colonne ordinate come cluster_ids
    cluster_ids: list[int]
    elapsed_s: float
    metrics: dict = field(default_factory=dict)
    cluster_silhouette: dict = field(default_factory=dict)


def soft_assignments(X: np.ndarray, centroids: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax sulle distanze quadratiche dai centroidi: temperatura bassa = assegnazioni più nette."""
    d2 = np.maximum((X * X).sum(1)[:, None] + (centroids * centroids).sum(1)[None, :] - 2 * X @ centroids.T, 0)
    logits = -d2 / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def default_hdbscan_params(n_samples: int) -> dict:
    min_cluster_size = max(5, min(200, int(math.sqrt(n_samples)) // 2))
    return {"min_cluster_size": min_cluster_size, "min_samples": max(1, min_cluster_size // 2)}


def run_clustering(X: np.ndarray, algorithm: str, params: dict) -> ClusteringResult:
    """X già preprocessata. `params` dipende dall'algoritmo (vedi GUI)."""
    start = time.perf_counter()
    seed = int(params.get("seed", 42))
    probs = None
    log.info("Clustering %s su %s con %s", algorithm, X.shape, params)

    if algorithm == "kmeans":
        k = int(params["k"])
        if not 1 < k <= len(X):
            raise ValueError(f"k deve essere tra 2 e {len(X)}")
        if params.get("minibatch"):
            model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=int(params.get("n_init", 3)),
                                    batch_size=2048)
        else:
            model = KMeans(n_clusters=k, random_state=seed, n_init=int(params.get("n_init", 10)))
        labels = model.fit_predict(X)
        ids = list(range(k))
        probs = soft_assignments(X, model.cluster_centers_, float(params.get("soft_temperature", 0.05)))

    elif algorithm == "hdbscan":
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=int(params["min_cluster_size"]),
            min_samples=int(params["min_samples"]) if params.get("min_samples") else None,
            cluster_selection_method=params.get("cluster_selection_method", "eom"),
            cluster_selection_epsilon=float(params.get("cluster_selection_epsilon", 0.0)),
            metric="euclidean", prediction_data=True, core_dist_n_jobs=-1,
        )
        labels = model.fit_predict(X)
        ids = cluster_ids(labels)
        if ids:
            try:
                probs = hdbscan.all_points_membership_vectors(model)
                probs = probs.reshape(len(X), -1) if probs.ndim == 1 else probs
                # HDBSCAN lascia una quota di "appartenenza al rumore": normalizziamo a distribuzione su K.
                probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
            except Exception as e:  # noqa: BLE001
                log.warning("Membership soft HDBSCAN non disponibile: %s", e)
    else:
        raise ValueError(f"Algoritmo sconosciuto: {algorithm}")

    labels = np.asarray(labels, dtype=np.int64)
    metrics, per_cluster = compute_metrics(X, labels, probs, seed=seed)
    result = ClusteringResult(
        algorithm=algorithm, params=params, labels=labels, probs=probs, cluster_ids=[int(c) for c in ids],
        elapsed_s=time.perf_counter() - start, metrics=metrics, cluster_silhouette=per_cluster,
    )
    log.info("Clustering %s completato in %.1fs: %s", algorithm, result.elapsed_s, metrics)
    return result


# ----------------------------------------------------------------------------- metriche
def compute_metrics(X: np.ndarray, labels: np.ndarray, probs: np.ndarray | None = None,
                    seed: int = 42, max_silhouette: int = 5000) -> tuple[dict, dict[int, float]]:
    """Metriche essenziali + silhouette per cluster (una sola passata di silhouette su un campione).

    - silhouette ∈ [-1, 1] (più alto = cluster compatti e separati)
    - davies_bouldin ≥ 0 (più basso è meglio)
    - noise_frac: punti non assegnati (solo HDBSCAN)
    - ambiguous_frac: punti con probabilità massima < 0.5 (casi di confine dalle assegnazioni soft)
    """
    valid = np.where(labels != -1)[0]
    ids, counts = np.unique(labels[valid], return_counts=True)
    m: dict = {
        "n_clusters": int(len(ids)),
        "noise_frac": float(1 - len(valid) / len(labels)) if len(labels) else 0.0,
        "min_size": int(counts.min()) if len(counts) else 0,
        "max_size": int(counts.max()) if len(counts) else 0,
        "silhouette": None, "davies_bouldin": None, "ambiguous_frac": None,
    }
    per_cluster: dict[int, float] = {}
    if len(ids) >= 2 and len(valid) > len(ids):
        rng = np.random.default_rng(seed)
        idx = valid if len(valid) <= max_silhouette else rng.choice(valid, max_silhouette, replace=False)
        try:
            s = silhouette_samples(X[idx], labels[idx])
            m["silhouette"] = float(s.mean())
            per_cluster = {int(c): float(s[labels[idx] == c].mean()) for c in np.unique(labels[idx])}
            m["davies_bouldin"] = float(davies_bouldin_score(X[valid], labels[valid]))
        except ValueError as e:
            log.warning("Metriche non calcolabili: %s", e)
    if probs is not None and probs.size:
        m["ambiguous_frac"] = float((probs.max(axis=1) < 0.5).mean())
    return m, per_cluster


def k_sweep(X: np.ndarray, k_values: list[int], seed: int = 42, workers: int = 4, progress=None) -> list[dict]:
    """Valuta più k con K-Means (inerzia per il gomito + silhouette), con più k calcolati in parallelo."""
    def one(k: int) -> dict:
        start = time.perf_counter()
        if len(X) > 20_000:
            model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
        else:
            model = KMeans(n_clusters=k, random_state=seed, n_init=4)
        labels = model.fit_predict(X)
        met, _ = compute_metrics(X, labels, seed=seed, max_silhouette=3000)
        return {"k": k, "inerzia": float(model.inertia_), "silhouette": met["silhouette"],
                "secondi": time.perf_counter() - start}

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, row in enumerate(pool.map(one, k_values), start=1):
            rows.append(row)
            if progress:
                progress(i, len(k_values))
    return rows


# ----------------------------------------------------------------------------- esempi e parole chiave
def cluster_ids(labels: np.ndarray) -> list[int]:
    """Id dei cluster presenti, in ordine, escluso il rumore (-1)."""
    return sorted(int(c) for c in set(labels.tolist()) - {-1})


def _by_centroid_distance(X: np.ndarray, labels: np.ndarray, cluster_id: int) -> np.ndarray:
    """Indici dei punti del cluster, dal più vicino al più lontano dal centroide."""
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return idx
    pts = X[idx]
    return idx[np.argsort(np.linalg.norm(pts - pts.mean(axis=0), axis=1))]


def representative_indices(X: np.ndarray, labels: np.ndarray, cluster_id: int,
                           per_tercile: int = 3, seed: int = 42) -> np.ndarray:
    """Indici di esempi rappresentativi: campionati dai terzili di distanza dal centroide
    (vicini, intermedi, periferici), così da mostrare anche la varietà interna del cluster."""
    ordered = _by_centroid_distance(X, labels, cluster_id)
    if len(ordered) == 0:
        return ordered
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.choice(t, size=min(per_tercile, len(t)), replace=False)
        for t in np.array_split(ordered, 3) if len(t)
    ])


def central_indices(X: np.ndarray, labels: np.ndarray, cluster_id: int, n: int = 5) -> np.ndarray:
    return _by_centroid_distance(X, labels, cluster_id)[:n]


def boundary_indices(probs: np.ndarray | None, labels: np.ndarray, cluster_id: int, n: int = 5) -> np.ndarray:
    """Punti del cluster con la confidenza più bassa (casi di confine)."""
    if probs is None:
        return np.empty(0, dtype=int)
    idx = np.where(labels == cluster_id)[0]
    return idx[np.argsort(probs[idx].max(axis=1))[:n]]


def nearest_clusters(X: np.ndarray, labels: np.ndarray, ids: list[int], n: int = 2) -> dict[int, list[int]]:
    """Per ogni cluster, i cluster con il centroide più vicino (usati per nomi contrastivi)."""
    if len(ids) < 2:
        return {c: [] for c in ids}
    C = np.vstack([X[labels == c].mean(axis=0) for c in ids])
    d = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return {c: [ids[j] for j in np.argsort(d[i])[:n]] for i, c in enumerate(ids)}


_EXTRA_STOP = {"good", "like", "just", "great", "really", "ve", "don", "did", "does", "didn", "doesn", "isn",
               "br", "quot", "lot", "bit", "pretty", "use"}
_TAG_RE = re.compile(r"<[^>]+>|https?://\S+")


def cluster_keywords(texts: list[str], labels: np.ndarray, top_n: int = 8) -> dict[int, list[str]]:
    """Parole chiave per cluster con c-TF-IDF (Grootendorst, BERTopic):
    W(t, c) = tf(t, c) / |c|  *  log(1 + A / f(t)),  A = media parole per cluster, f(t) = frequenza totale."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    ids = cluster_ids(labels)
    if not ids:
        return {}
    clean =[_TAG_RE.sub(" ", t) for t in texts]
    vec = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS | _EXTRA_STOP), ngram_range=(1, 2),
                          min_df=2 if len(texts) < 200 else 5, max_features=30_000,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    try:
        counts = vec.fit_transform(clean)
    except ValueError:  # vocabolario vuoto (pochi testi)
        return {c: [] for c in ids}
    vocab = np.array(vec.get_feature_names_out())
    tf = np.vstack([np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in ids]).astype(np.float64)
    A = tf.sum() / len(ids)
    w = (tf / np.clip(tf.sum(axis=1, keepdims=True), 1, None)) * np.log1p(A / np.clip(tf.sum(axis=0), 1, None))
    return {c: vocab[np.argsort(w[row])[::-1][:top_n]].tolist() for row, c in enumerate(ids)}
