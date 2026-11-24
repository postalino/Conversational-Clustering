import numpy as np
import pytest

import runs
from embeddings import EmbeddingStore


def test_embedding_store_chunks_and_compaction(tmp_path):
    store = EmbeddingStore("owner/ds", "model-a", root=tmp_path)
    store.append([3, 1], [[1, 0], [0, 1]])
    store.append([2], [[1, 1]])
    assert store.count() == 3 and store.missing_ids([1, 2, 5]) == [5]
    np.testing.assert_array_equal(store.get([2, 3, 1]), [[1, 1], [1, 0], [0, 1]])
    store.compact()
    np.testing.assert_array_equal(store.get([2, 3, 1]), [[1, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError):
        store.append([9], [[1, 2, 3]])  # dimensione diversa
    assert EmbeddingStore("owner/ds", "model-b", root=tmp_path).count() == 0  # modelli separati


def test_runs_save_load_select(tmp_path):
    run = runs.Run(id="r1", name="test", created="now", dataset="ds", embedding_model="m", sample={"n": 3},
                   preprocess={}, algorithm="kmeans", params={"k": 2}, metrics={"n_clusters": 2},
                   clusters={"0": {"name": "A"}}, ids=np.array([1, 2, 3]), labels=np.array([0, 1, 1]),
                   probs=np.full((3, 2), 0.5))
    runs.save_run(run, root=tmp_path)
    loaded = runs.load_run("r1", root=tmp_path)
    assert loaded.clusters["0"]["name"] == "A"
    np.testing.assert_array_equal(loaded.labels, [0, 1, 1])
    runs.set_selected("r1", root=tmp_path)
    assert runs.get_selected(root=tmp_path) == "r1"
    runs.delete_run("r1", root=tmp_path)
    assert runs.get_selected(root=tmp_path) is None


def test_compute_missing_keeps_finished_batches_on_error(tmp_path):
    """Se un batch fallisce, quelli riusciti restano in cache e il progresso avanza durante il calcolo."""
    import threading

    import pandas as pd

    from embeddings import compute_missing

    class FlakyService:
        model = "fake"
        lock = threading.Lock()
        calls = 0

        def embed_texts(self, texts):
            import time

            time.sleep(0.02)  # latenza simulata: i batch terminano in momenti diversi, come con un'API vera
            with self.lock:
                self.calls += 1
                n = self.calls
            if n == 5:
                raise RuntimeError("429 rate limit")
            return [[1.0, 0.0] for _ in texts]

    store = EmbeddingStore("ds", "fake", root=tmp_path)
    df = pd.DataFrame({"Id": range(100), "Text": ["t"] * 100})
    updates = []
    with pytest.raises(RuntimeError, match="429"):
        compute_missing(FlakyService(), store, df, batch_size=10, workers=1, flush_every=20,
                        progress=lambda d, t: updates.append(d))
    # tutti i batch riusciti sono salvati; manca solo il batch fallito (il 5°: id 40-49)
    missing = set(store.missing_ids(range(100)))
    assert set(range(40, 50)) <= missing and store.count() >= 40
    assert all(i >= 50 for i in missing - set(range(40, 50)))  # eventuali altri mancanti: solo batch annullati
    assert len(updates) >= 4  # avanzamento aggiornato batch per batch, non solo alla fine
    healthy = FlakyService()
    healthy.calls = -1000  # non raggiunge mai la chiamata che fallisce
    done = compute_missing(healthy, store, df, batch_size=10, workers=4)  # ripresa: solo i mancanti
    assert store.count() == 100 and done == len(missing)


def test_store_is_keyed_by_text_column_and_refreshes_index(tmp_path):
    a = EmbeddingStore("ds", "m", root=tmp_path)  # colonna "Text" (cartella storica)
    b = EmbeddingStore("ds", "m", root=tmp_path, text_col="Summary")
    a.append([1], [[1.0, 0.0]])
    assert b.missing_ids([1]) == [1]  # stessi id ma altra colonna: niente vettori riusati
    reader = EmbeddingStore("ds", "m", root=tmp_path)
    assert reader.count() == 1
    a.append([2], [[0.0, 1.0]])
    a.compact()
    np.testing.assert_array_equal(reader.get([2]), [[0.0, 1.0]])  # indice ricaricato dopo scrittura/compattazione
