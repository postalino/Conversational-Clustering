from clustering import get_representative_examples
from agent_actions import rename_cluster, name_cluster
import numpy as np

# Rename
def rename_cluster_state(
    llm_service,
    data,
    cluster_metadata: dict,
    cluster_id,
    reason: str | None = None,
) -> dict:
    if cluster_id not in cluster_metadata:
        raise KeyError(f"cluster_id {cluster_id!r} not found in cluster_metadata")

    examples = get_representative_examples(
        data=data,
        cluster_id=cluster_id,
        samples_per_tercile=2,
    )

    result = rename_cluster(
        llm_service=llm_service,
        examples=examples,
        old_name=cluster_metadata[cluster_id]["name"],
        reason=reason,
    )

    cluster_metadata[cluster_id] = result
    return result

# Merge 
def binary_merge_cluster_state(
    llm_service,
    data,
    cluster_metadata: dict,
    cluster_id_1,
    cluster_id_2,
) -> dict:
    if cluster_id_1 not in cluster_metadata:
        raise KeyError(f"cluster_id_1 {cluster_id_1!r} not found in cluster_metadata")
    if cluster_id_2 not in cluster_metadata:
        raise KeyError(f"cluster_id_2 {cluster_id_2!r} not found in cluster_metadata")

    new_cluster_id = min(cluster_id_1, cluster_id_2)
    old_cluster_id = max(cluster_id_1, cluster_id_2)

    data.loc[data["cluster_id"] == old_cluster_id, "cluster_id"] = new_cluster_id

    if old_cluster_id in cluster_metadata:
        del cluster_metadata[old_cluster_id]

    examples = get_representative_examples(
        data=data,
        cluster_id=new_cluster_id,
        samples_per_tercile=2,
    )

    result = name_cluster(
        llm_service=llm_service,
        examples=examples,
    )

    cluster_metadata[new_cluster_id] = result
    return result

# Split
def split_cluster_state(
    llm_service,
    data,
    cluster_metadata: dict,
    cluster_id,
) -> dict:
    if cluster_id not in cluster_metadata:
        raise KeyError(f"cluster_id {cluster_id!r} not found in cluster_metadata")

    cluster_rows = data[data["cluster_id"] == cluster_id]

    if len(cluster_rows) < 2:
        raise ValueError("Cannot split a cluster with fewer than 2 items")

    cluster_embeddings = np.vstack(cluster_rows["embedding"].to_numpy())

    from clustering import cluster_with_kmeans
    new_labels = cluster_with_kmeans(cluster_embeddings, k=2)

    new_cluster_id_1 = max(cluster_metadata.keys()) + 1
    new_cluster_id_2 = new_cluster_id_1 + 1

    cluster_indices = cluster_rows.index.to_numpy()

    indices_1 = cluster_indices[new_labels == 0]
    indices_2 = cluster_indices[new_labels == 1]

    data.loc[indices_1, "cluster_id"] = new_cluster_id_1
    data.loc[indices_2, "cluster_id"] = new_cluster_id_2

    del cluster_metadata[cluster_id]

    examples_1 = get_representative_examples(
        data=data,
        cluster_id=new_cluster_id_1,
        samples_per_tercile=2,
    )

    examples_2 = get_representative_examples(
        data=data,
        cluster_id=new_cluster_id_2,
        samples_per_tercile=2,
    )

    result_1 = name_cluster(
        llm_service=llm_service,
        examples=examples_1,
    )

    result_2 = name_cluster(
        llm_service=llm_service,
        examples=examples_2,
    )

    cluster_metadata[new_cluster_id_1] = result_1
    cluster_metadata[new_cluster_id_2] = result_2

    return {
        new_cluster_id_1: result_1,
        new_cluster_id_2: result_2,
    }