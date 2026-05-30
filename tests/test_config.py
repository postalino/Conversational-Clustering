import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config


def test_load_env_with_required_keys_present(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KAGGLE_USERNAME=test_user\nKAGGLE_KEY=test_key\n", encoding="utf-8"
    )

    monkeypatch.setattr(config, "ENV_PATH", env_file)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    values = config.load_env(required_keys=["KAGGLE_USERNAME", "KAGGLE_KEY"])

    assert values["KAGGLE_USERNAME"] == "test_user"
    assert values["KAGGLE_KEY"] == "test_key"


def test_load_env_with_missing_required_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("KAGGLE_USERNAME=test_user\n", encoding="utf-8")

    monkeypatch.setattr(config, "ENV_PATH", env_file)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with pytest.raises(RuntimeError, match="KAGGLE_KEY"):
        config.load_env(required_keys=["KAGGLE_USERNAME", "KAGGLE_KEY"])
