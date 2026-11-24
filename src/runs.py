"""Persistenza dei "set iniziali" (run di clustering) e scelta di quello da usare con l'agente.

    data/runs/<run_id>/run.json     configurazione completa, metriche, nomi dei cluster
    data/runs/<run_id>/arrays.npz   ids, labels, probs (opzionale), coords 2D (opzionale)
    data/runs/selected.json         run scelto come set iniziale
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from config import RUNS_DIR
from logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Run:
    id: str
    name: str
    created: str
    dataset: str
    embedding_model: str
    sample: dict
    preprocess: dict
    algorithm: str
    params: dict
    metrics: dict
    clusters: dict = field(default_factory=dict)  # str(cluster_id) -> {name, description, evidence, keywords, size}
    notes: str = ""
    labels_model: str | None = None  # modello LLM usato per i nomi
    prompt_versions: dict | None = None  # versione (hash) dei prompt usati per i nomi

    # array (non serializzati nel json)
    ids: np.ndarray | None = None
    labels: np.ndarray | None = None
    probs: np.ndarray | None = None
    coords: np.ndarray | None = None

    def meta(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("ids", "labels", "probs", "coords")}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def new_run_id(algorithm: str) -> str:
    # suffisso casuale: due salvataggi nello stesso secondo non si sovrascrivono
    return f"{datetime.now():%Y%m%d-%H%M%S}-{algorithm}-{uuid.uuid4().hex[:4]}"


def save_run(run: Run, root: Path = RUNS_DIR) -> Path:
    d = root / run.id
    d.mkdir(parents=True, exist_ok=False)  # mai sovrascrivere un set esistente
    update_meta(run, root)
    arrays = {"ids": run.ids, "labels": run.labels}
    if run.probs is not None:
        arrays["probs"] = run.probs.astype(np.float32)
    if run.coords is not None:
        arrays["coords"] = run.coords.astype(np.float32)
    np.savez_compressed(d / "arrays.npz", **arrays)
    log.info("Set iniziale salvato: %s (%s)", run.id, run.name)
    return d


def update_meta(run: Run, root: Path = RUNS_DIR) -> None:
    (root / run.id / "run.json").write_text(
        json.dumps(run.meta(), indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")


def load_arrays(run_id: str, root: Path = RUNS_DIR) -> dict[str, np.ndarray | None]:
    """Array di un set (ids, labels, probs, coords). Non cambiano mai dopo `save_run`:
    `update_meta` riscrive solo run.json, quindi possono essere tenuti in cache."""
    with np.load(root / run_id / "arrays.npz") as a:
        return {k: (a[k] if k in a else None) for k in ("ids", "labels", "probs", "coords")}


def load_run(run_id: str, with_arrays: bool = True, root: Path = RUNS_DIR, arrays: dict | None = None) -> Run:
    """`arrays`: array già caricati (es. dalla cache della GUI) invece di rileggerli dal disco."""
    meta = json.loads((root / run_id / "run.json").read_text(encoding="utf-8"))
    run = Run(**meta)
    if with_arrays:
        a = arrays if arrays is not None else load_arrays(run_id, root)
        run.ids, run.labels, run.probs, run.coords = a["ids"], a["labels"], a["probs"], a["coords"]
    return run


def list_runs(root: Path = RUNS_DIR) -> list[Run]:
    if not root.exists():
        return []
    runs = []
    for d in sorted(root.iterdir(), reverse=True):
        if (d / "run.json").exists():
            try:
                runs.append(load_run(d.name, with_arrays=False, root=root))
            except Exception as e:  # noqa: BLE001
                log.warning("Run illeggibile %s: %s", d.name, e)
    return runs


def delete_run(run_id: str, root: Path = RUNS_DIR) -> None:
    if get_selected(root) == run_id:
        (root / "selected.json").unlink(missing_ok=True)
    shutil.rmtree(root / run_id)
    log.info("Set iniziale eliminato: %s", run_id)


def set_selected(run_id: str | None, root: Path = RUNS_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "selected.json"
    if run_id is None:
        path.unlink(missing_ok=True)
        log.info("Nessun set iniziale selezionato")
        return
    path.write_text(json.dumps({"run_id": run_id, "selected_at": datetime.now().isoformat(timespec="seconds")}),
                    encoding="utf-8")
    log.info("Set iniziale selezionato: %s", run_id)


def get_selected(root: Path = RUNS_DIR) -> str | None:
    path = root / "selected.json"
    if not path.exists():
        return None
    run_id = json.loads(path.read_text(encoding="utf-8")).get("run_id")
    return run_id if run_id and (root / run_id).exists() else None


def agreement(a: Run, b: Run) -> dict | None:
    """ARI / NMI tra due run sugli elementi in comune."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < 2:
        return None
    la, lb = a.labels[ia], b.labels[ib]
    return {"common": int(len(common)), "ari": float(adjusted_rand_score(la, lb)),
            "nmi": float(normalized_mutual_info_score(la, lb))}
