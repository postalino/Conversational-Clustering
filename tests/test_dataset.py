import sys
import types
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dataset


def _install_fake_kaggle(monkeypatch):
    instances = []

    class FakeKaggleApi:
        def __init__(self):
            self.authenticated = False
            self.download_calls = []
            instances.append(self)

        def authenticate(self):
            self.authenticated = True

        def dataset_download_files(self, dataset, path, unzip):
            self.download_calls.append(
                {
                    "dataset": dataset,
                    "path": path,
                    "unzip": unzip,
                }
            )

    kaggle_module = types.ModuleType("kaggle")
    kaggle_api_module = types.ModuleType("kaggle.api")
    kaggle_api_extended_module = types.ModuleType("kaggle.api.kaggle_api_extended")
    kaggle_api_extended_module.KaggleApi = FakeKaggleApi

    monkeypatch.setitem(sys.modules, "kaggle", kaggle_module)
    monkeypatch.setitem(sys.modules, "kaggle.api", kaggle_api_module)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", kaggle_api_extended_module)

    return instances


def test_dataset_downloader_init_with_valid_env(monkeypatch):
    monkeypatch.setattr(
        dataset,
        "load_env",
        lambda required_keys: {
            "KAGGLE_USERNAME": "user",
            "KAGGLE_KEY": "key",
            "KAGGLE_DATASET": "owner/my-dataset",
        },
    )
    instances = _install_fake_kaggle(monkeypatch)

    downloader = dataset.DatasetDownloader()

    assert downloader.dataset_name == "owner/my-dataset"
    assert instances
    assert downloader.api is instances[0]
    assert instances[0].authenticated is True


def test_dataset_downloader_download_data_creates_dir_and_calls_api(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "load_env",
        lambda required_keys: {
            "KAGGLE_USERNAME": "user",
            "KAGGLE_KEY": "key",
            "KAGGLE_DATASET": "owner/my-dataset",
        },
    )
    instances = _install_fake_kaggle(monkeypatch)

    target_dir = tmp_path / "data"
    downloader = dataset.DatasetDownloader(data_dir=target_dir, unzip=False)

    downloader.download_data()

    assert target_dir.exists()
    assert len(instances[0].download_calls) == 1
    call = instances[0].download_calls[0]
    assert call["dataset"] == "owner/my-dataset"
    assert call["path"] == str(target_dir)
    assert call["unzip"] is False
