import sys
import types
from pathlib import Path

import pandas as pd

import dataset


def test_download_is_lazy_and_idempotent(tmp_path, monkeypatch):
    """Kaggle si autentica solo quando serve; un secondo download non riscarica."""
    calls = []

    class FakeKaggleApi:
        def authenticate(self):
            calls.append("auth")

        def dataset_download_files(self, dataset, path, unzip):
            calls.append("download")
            pd.DataFrame({"Id": [1, 2], "Text": ["a b c", "d e f"]}).to_csv(Path(path) / "r.csv", index=False)

    mod = types.ModuleType("kaggle.api.kaggle_api_extended")
    mod.KaggleApi = FakeKaggleApi
    for name, m in (("kaggle", types.ModuleType("kaggle")), ("kaggle.api", types.ModuleType("kaggle.api")),
                    ("kaggle.api.kaggle_api_extended", mod)):
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setattr(dataset, "load_env", lambda required_keys: {})

    dl = dataset.DatasetDownloader(data_dir=tmp_path, dataset_name="owner/ds")
    assert calls == []
    dl.download_data()
    dl.download_data()
    assert calls == ["auth", "download"]
    assert dl.metadata()["dataset"] == "owner/ds"
    assert len(dl.load_raw()) == 2 and dl.parquet_path.exists()


def test_make_sample_filters_and_is_reproducible():
    raw = pd.DataFrame({"Id": range(10), "Text": ["short", "a long enough text"] * 5, "Score": range(10)})
    out = dataset.make_sample(raw, dataset.SampleConfig(sample_size=None, min_chars=10, dedupe=True))
    assert list(out.columns) == ["Id", "Text", "Score"] and len(out) == 1

    cfg = dataset.SampleConfig(sample_size=3, min_chars=0, dedupe=False, seed=1)
    assert dataset.make_sample(raw, cfg)["Id"].tolist() == dataset.make_sample(raw, cfg)["Id"].tolist()
    only = dataset.make_sample(raw, dataset.SampleConfig(sample_size=None, min_chars=0, dedupe=False),
                               allowed_ids=[2, 4])
    assert sorted(only["Id"]) == [2, 4]


def test_make_sample_rejects_invalid_id_columns():
    import pytest

    raw = pd.DataFrame({"Id": [1, 2, 2], "Code": ["a", "b", "c"], "Text": ["x y z"] * 3})
    with pytest.raises(ValueError, match="interi"):
        dataset.make_sample(raw, dataset.SampleConfig(id_col="Code", min_chars=0, dedupe=False))
    with pytest.raises(ValueError, match="ripetuti"):
        dataset.make_sample(raw, dataset.SampleConfig(id_col="Id", min_chars=0, dedupe=False))
