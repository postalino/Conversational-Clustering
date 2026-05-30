import math
import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples


def get_hdbscan_params(n_samples: int) -> dict:
    min_cluster_size = max(
        3,
        min(50, max(5, int(math.sqrt(n_samples)))) // 2,
    )
    min_samples = max(1, min_cluster_size // 2)

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }


def cluster_with_hdbscan(embeddings: np.ndarray) -> np.ndarray:
    params = get_hdbscan_params(len(embeddings))

    model = hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
    )

    return model.fit_predict(embeddings)


def cluster_with_kmeans(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k deve essere maggiore di 0")

    if k > len(embeddings):
        raise ValueError("k non può essere maggiore del numero di elementi")

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init="auto",
    )

    return model.fit_predict(embeddings)


def get_representative_examples(
    data,
    cluster_id: int,
    samples_per_tercile: int = 3,
    seed: int = 42,
) -> list[dict]:
    cluster_data = data[data["cluster_id"] == cluster_id]

    if cluster_data.empty:
        return []

    cluster_embeddings = np.vstack(cluster_data["embedding"].to_numpy())
    centroid = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    terciles = np.array_split(np.argsort(distances), 3)
    rng = np.random.default_rng(seed)

    selected = np.concatenate([
        rng.choice(t, size=min(samples_per_tercile, len(t)), replace=False)
        for t in terciles
        if len(t) > 0
    ])

    examples_df = cluster_data.iloc[selected]

    return examples_df[["Id", "Text"]].to_dict(orient="records")


def calculate_cluster_cohesion(data, cluster_id: int) -> float:
    cluster_data = data[data["cluster_id"] == cluster_id]

    if len(cluster_data) <= 1:
        return 0.0

    cluster_embeddings = np.vstack(cluster_data["embedding"].to_numpy())
    centroid = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    return float(distances.mean())


def calculate_cluster_silhouette(data, cluster_id: int) -> float:
    if "cluster_id" not in data.columns:
        raise ValueError("data must contain a 'cluster_id' column")

    labels = data["cluster_id"].to_numpy()

    if len(set(labels)) < 2:
        return 0.0

    cluster_mask = data["cluster_id"] == cluster_id

    if cluster_mask.sum() <= 1:
        return 0.0

    embeddings = np.vstack(data["embedding"].to_numpy())
    sample_scores = silhouette_samples(embeddings, labels, metric="euclidean")

    return float(sample_scores[cluster_mask].mean())


def calculate_all_davies_bouldin(data) -> dict:
    if "cluster_id" not in data.columns:
        raise ValueError("data must contain a 'cluster_id' column")

    valid_data = data[data["cluster_id"] != -1]
    cluster_ids = valid_data["cluster_id"].unique()

    if len(cluster_ids) < 2:
        return {}

    embeddings = np.vstack(valid_data["embedding"].to_numpy())
    labels = valid_data["cluster_id"].to_numpy()

    result = {}
    for cluster_id in cluster_ids:
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() <= 1:
            result[cluster_id] = 0.0
            continue

        cluster_indices = np.where(cluster_mask)[0]
        other_mask = ~cluster_mask
        other_labels = labels[other_mask]
        if len(np.unique(other_labels)) < 1:
            result[cluster_id] = 0.0
            continue

        cluster_embeddings = embeddings[cluster_mask]
        other_embeddings = embeddings[other_mask]
        combined_embeddings = np.vstack([cluster_embeddings, other_embeddings])
        combined_labels = np.concatenate([
            np.zeros(len(cluster_embeddings), dtype=int),
            np.ones(len(other_embeddings), dtype=int),
        ])

        try:
            result[cluster_id] = float(davies_bouldin_score(combined_embeddings, combined_labels))
        except ValueError:
            result[cluster_id] = 0.0

    return result
