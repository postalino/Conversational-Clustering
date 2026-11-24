"""Download del dataset da Kaggle e costruzione di campioni riproducibili."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DATA_DIR, get_env, load_env
from logging_setup import get_logger

log = get_logger(__name__)


class DatasetDownloader:
    """Scarica un dataset Kaggle in `data/dataset.csv` (+ `data/dataset.json` con i metadati)."""

    def __init__(self, data_dir: Path | None = None, dataset_name: str | None = None) -> None:
        self.dataset_name = dataset_name or get_env("KAGGLE_DATASET")
        self.data_dir = data_dir or DATA_DIR
        self.download_dir = self.data_dir / "_tmp_dataset"
        self.csv_path = self.data_dir / "dataset.csv"
        self.meta_path = self.data_dir / "dataset.json"
        self._api = None

    @property
    def api(self):
        """Kaggle client, authenticated only when actually needed (download)."""
        if self._api is None:
            load_env(required_keys=["KAGGLE_USERNAME", "KAGGLE_KEY"])
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            self._api = api
        return self._api

    def is_downloaded(self) -> bool:
        return self.csv_path.exists()

    def metadata(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.csv_path.exists():  # CSV scaricato con la versione precedente del progetto
            return {"dataset": get_env("KAGGLE_DATASET") or "local", "file": self.csv_path.name}
        return {}

    @property
    def slug(self) -> str:
        return self.metadata().get("dataset") or self.dataset_name or "local"

    def download_data(self, force: bool = False) -> Path:
        if not self.dataset_name:
            raise ValueError("Nessun dataset Kaggle indicato (KAGGLE_DATASET o campo nella GUI)")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists() and not force and self.metadata().get("dataset") == self.dataset_name:
            log.info("Dataset già presente: %s", self.csv_path)
            return self.csv_path

        if self.download_dir.exists():
            shutil.rmtree(self.download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        try:
            log.info("Download da Kaggle: %s", self.dataset_name)
            self.api.dataset_download_files(dataset=self.dataset_name, path=str(self.download_dir), unzip=True)

            csv_files = list(self.download_dir.rglob("*.csv"))
            if not csv_files:
                raise ValueError(f"Nessun file CSV trovato nel dataset {self.dataset_name!r}")
            if len(csv_files) > 1:
                csv_files.sort(key=lambda p: p.stat().st_size, reverse=True)
                log.warning("Trovati %d CSV, uso il più grande: %s", len(csv_files), csv_files[0].name)

            # copyfile (non copy2): il CSV deve avere una data nuova, altrimenti cache basate sulla data di
            # modifica (copia Parquet, cache della GUI) potrebbero considerarlo invariato.
            shutil.copyfile(csv_files[0], self.csv_path)
            self.parquet_path.unlink(missing_ok=True)  # la copia Parquet era del dataset precedente
            self.meta_path.write_text(json.dumps({
                "dataset": self.dataset_name,
                "file": csv_files[0].name,
                "downloaded": datetime.now().isoformat(timespec="seconds"),
            }, indent=2), encoding="utf-8")
            log.info("Dataset salvato in %s", self.csv_path)
            return self.csv_path
        finally:
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)

    @property
    def parquet_path(self) -> Path:
        return self.csv_path.with_suffix(".parquet")

    def load_raw(self) -> pd.DataFrame:
        """Legge il dataset. Alla prima lettura converte il CSV in Parquet (colonnare, compresso):
        le letture successive sono molto più rapide."""
        if not self.csv_path.exists():
            raise FileNotFoundError("Dataset non ancora scaricato")
        pq = self.parquet_path
        if pq.exists() and pq.stat().st_mtime >= self.csv_path.stat().st_mtime:
            try:
                return pd.read_parquet(pq)
            except Exception as e:  # noqa: BLE001 - Parquet corrotto: si rigenera dal CSV
                log.warning("Parquet illeggibile, rigenero dal CSV: %s", e)
        df = pd.read_csv(self.csv_path, engine="pyarrow")
        try:
            df.to_parquet(pq, index=False)
            log.info("Creata copia Parquet del dataset: %s", pq.name)
        except Exception as e:  # noqa: BLE001 - la cache è un'ottimizzazione, non un requisito
            log.warning("Impossibile scrivere il Parquet: %s", e)
        return df


@dataclass
class SampleConfig:
    text_col: str = "Text"
    id_col: str | None = "Id"  # None -> indice di riga
    sample_size: int | None = 2000
    seed: int = 42
    min_chars: int = 20
    max_chars: int | None = None
    dedupe: bool = True
    extra_cols: list[str] = field(default_factory=lambda: ["Summary", "Score"])
    only_cached: bool = False  # campiona solo tra i testi con embedding già in cache

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleConfig":
        """Ignora le chiavi sconosciute (es. "n" salvato nei set)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def unfiltered(self) -> "SampleConfig":
        """Stesse colonne, nessun filtro né campionamento: serve a ritrovare testi per id."""
        return replace(self, sample_size=None, min_chars=0, max_chars=None, dedupe=False, only_cached=False)


def make_sample(raw: pd.DataFrame, cfg: SampleConfig, allowed_ids=None) -> pd.DataFrame:
    """Returns a DataFrame with columns Id (int), Text (str) + any extra columns present.

    `allowed_ids`: se indicato, il campione è estratto solo tra questi id (es. quelli già in cache)."""
    if cfg.text_col not in raw.columns:
        raise KeyError(f"Colonna testo {cfg.text_col!r} non presente. Colonne: {list(raw.columns)}")
    if cfg.id_col and cfg.id_col not in raw.columns:
        raise KeyError(f"Colonna id {cfg.id_col!r} non presente. Colonne: {list(raw.columns)}")
    if cfg.id_col:
        ids = raw[cfg.id_col]
        if not pd.api.types.is_integer_dtype(ids):
            raise ValueError(f"La colonna id {cfg.id_col!r} deve contenere interi (tipo attuale: {ids.dtype}). "
                             "Usa un'altra colonna o l'indice di riga.")
        if ids.duplicated().any():
            raise ValueError(f"La colonna id {cfg.id_col!r} ha {int(ids.duplicated().sum()):,} valori ripetuti: "
                             "non identifica univocamente i testi. Usa un'altra colonna o l'indice di riga.")

    extras = [c for c in cfg.extra_cols if c in raw.columns and c not in (cfg.text_col, cfg.id_col)]
    df = pd.DataFrame({
        "Id": raw[cfg.id_col] if cfg.id_col else raw.index,
        "Text": raw[cfg.text_col],
    })
    for c in extras:
        df[c] = raw[c]

    n0 = len(df)
    df = df.dropna(subset=["Id", "Text"])
    df["Text"] = df["Text"].astype(str).str.strip()
    df["Id"] = pd.to_numeric(df["Id"], errors="coerce")
    df = df.dropna(subset=["Id"])
    df["Id"] = df["Id"].astype("int64")
    df = df.drop_duplicates(subset="Id")
    if allowed_ids is not None:
        df = df[df["Id"].isin(set(int(i) for i in allowed_ids))]

    lengths = df["Text"].str.len()
    mask = lengths >= cfg.min_chars
    if cfg.max_chars:
        mask &= lengths <= cfg.max_chars
    df = df[mask]
    if cfg.dedupe:
        df = df.drop_duplicates(subset="Text")

    if cfg.sample_size is not None and cfg.sample_size < len(df):
        df = df.sample(n=cfg.sample_size, random_state=cfg.seed)

    log.info("Campione: %d righe (da %d, filtri: min=%s max=%s dedupe=%s)",
             len(df), n0, cfg.min_chars, cfg.max_chars, cfg.dedupe)
    return df.reset_index(drop=True)
