import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"


def load_env(required_keys: Sequence[str] | None = None, override: bool = True) -> dict[str, str]:
	"""Load .env values into the process and optionally validate required keys."""
	load_dotenv(ENV_PATH, override=override)

	if not required_keys:
		return {}

	values: dict[str, str] = {}
	missing: list[str] = []

	for key in required_keys:
		value = os.getenv(key)
		if value:
			values[key] = value
		else:
			missing.append(key)

	if missing:
		missing_str = ", ".join(missing)
		raise RuntimeError(f"Missing required environment variables in .env: {missing_str}")

	return values

