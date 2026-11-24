"""Configurazione del progetto: percorsi, lettura/scrittura del file .env."""

import os
import re
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv, set_key

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
DATA_DIR = ROOT_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RUNS_DIR = DATA_DIR / "runs"
LOGS_DIR = ROOT_DIR / "logs"


def load_env(required_keys: Sequence[str] = ()) -> dict[str, str]:
    """Load .env values into the process (overriding) and validate the required keys."""
    load_dotenv(ENV_PATH, override=True)
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables in .env: {', '.join(missing)}")
    return {k: os.environ[k] for k in required_keys}


def get_env(key: str, default: str = "") -> str:
    load_dotenv(ENV_PATH, override=False)
    return os.getenv(key, default) or default


def save_env_values(values: dict[str, str]) -> None:
    """Persist the given keys into .env (creating it if needed) and into os.environ."""
    ENV_PATH.touch(exist_ok=True)
    for key, value in values.items():
        set_key(str(ENV_PATH), key, value, quote_mode="never")
        os.environ[key] = value


def slugify(value: str) -> str:
    """Filesystem-safe name, e.g. 'openai/text-embedding-3-large' -> 'openai__text-embedding-3-large'."""
    value = value.strip().replace("/", "__").replace(":", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value) or "default"


def mask_secret(value: str | None) -> str:
    if not value:
        return "(non impostata)"
    if len(value) <= 8:
        return "••••"
    return f"{value[:5]}…{value[-4:]}"
