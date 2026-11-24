import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering import cluster_keywords, run_clustering


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(600, 20, centers=4, cluster_std=1.0, random_state=0)
    return X.astype(np.float32), y


@pytest.mark.parametrize("algo,params", [
    ("kmeans", {"k": 4}),
    ("hdbscan", {"min_cluster_size": 20, "min_samples": 5}),
])
def test_algorithms_recover_blobs(blobs, algo, params):
    X, y = blobs
    res = run_clustering(X, algo, params)
    valid = res.labels != -1
    assert adjusted_rand_score(y[valid], res.labels[valid]) > 0.9
    assert res.metrics["n_clusters"] == 4 and res.metrics["silhouette"] > 0.5
    np.testing.assert_allclose(res.probs.sum(1)[valid], 1, atol=1e-3)  # soft = distribuzioni


def test_keywords_are_distinctive():
    texts = ["coffee beans roast", "coffee aroma beans", "dog food kibble", "dog treats food"]
    kw = cluster_keywords(texts, np.array([0, 0, 1, 1]), top_n=3)
    assert "coffee" in kw[0] and "dog" in kw[1]
