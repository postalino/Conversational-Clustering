"""Nomi e descrizioni dei cluster generati dall'LLM.

Due fasi:
1. `name_clusters`: una chiamata per cluster, **in parallelo**. Ogni prompt è indipendente dagli altri
   (contiene le parole chiave dei cluster vicini invece dei loro nomi), quindi non c'è ordine da rispettare.
2. `refine_names`: una sola chiamata che vede tutti i cluster e rende i nomi distinti e coerenti.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np

from clustering import central_indices, cluster_ids as non_noise_ids, nearest_clusters, representative_indices
from logging_setup import get_logger
from prompts import get_prompt

log = get_logger(__name__)

MAX_EXAMPLE_CHARS = 600

SCHEMA_NAME_CLUSTER = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "coherence": {"type": "integer"},
    },
    "required": ["name", "description", "evidence", "coherence"],
    "additionalProperties": False,
}

SCHEMA_REFINE = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["cluster_id", "name", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clusters"],
    "additionalProperties": False,
}


@dataclass
class NamingInput:
    cluster_id: int
    size: int
    share: float
    keywords: list[str]
    neighbor_keywords: dict[int, list[str]]
    central: list[str]
    diverse: list[str]


def _bullets(texts: list[str]) -> str:
    out = []
    for t in texts:
        t = " ".join(str(t).split())
        out.append(f"- {t[:MAX_EXAMPLE_CHARS]}{'…' if len(t) > MAX_EXAMPLE_CHARS else ''}")
    return "\n".join(out) or "- (none)"


def build_inputs(X: np.ndarray, labels: np.ndarray, texts: list[str], cluster_ids: list[int],
                 keywords: dict[int, list[str]], n_central: int = 4, per_tercile: int = 3) -> list[NamingInput]:
    """Prepara (in modo deterministico, senza LLM) il materiale per ogni cluster."""
    # vicini calcolati su *tutti* i cluster, anche se ne vengono nominati solo alcuni (o uno solo)
    neighbors = nearest_clusters(X, labels, non_noise_ids(labels), n=2)
    n = len(labels)
    inputs = []
    for c in cluster_ids:
        central = central_indices(X, labels, c, n=n_central)
        central_set = set(central)
        diverse = [i for i in representative_indices(X, labels, c, per_tercile=per_tercile, seed=42)
                   if i not in central_set]
        size = int((labels == c).sum())
        inputs.append(NamingInput(
            cluster_id=c, size=size, share=size / n, keywords=keywords.get(c, []),
            neighbor_keywords={o: keywords.get(o, [])[:6] for o in neighbors[c]},
            central=[texts[i] for i in central], diverse=[texts[i] for i in diverse],
        ))
    return inputs


def keywords_by_cluster(clusters: dict[str, dict]) -> dict[int, list[str]]:
    """Parole chiave dal dizionario dei cluster (chiavi stringa, come salvato nei set)."""
    return {int(c): v.get("keywords", []) for c, v in clusters.items()}


def naming_messages(item: NamingInput, language: str = "English") -> list[dict[str, str]]:
    """Messaggi inviati all'LLM per nominare un cluster (usati anche dall'anteprima della pagina Prompt)."""
    neighbors = "\n".join(f"- cluster {o}: {', '.join(kw) or '(none)'}" for o, kw in item.neighbor_keywords.items())
    return get_prompt("cluster_naming").messages(
        language=language, size=item.size, share=f"{item.share:.1%}",
        keywords=", ".join(item.keywords) or "(none)", neighbors=neighbors or "- (none)",
        central=_bullets(item.central), diverse=_bullets(item.diverse),
    )


def name_cluster(llm_service, item: NamingInput, language: str = "English") -> dict:
    result = llm_service.chat_json(naming_messages(item, language), schema=SCHEMA_NAME_CLUSTER, temperature=0.2)
    for key in ("name", "description"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Risposta LLM senza campo '{key}': {result}")
    result["evidence"] = [str(e) for e in result.get("evidence") or []][:3]
    try:
        result["coherence"] = int(min(5, max(1, int(result.get("coherence", 3)))))
    except (TypeError, ValueError):
        result["coherence"] = None
    return result


def name_clusters(llm_service, inputs: list[NamingInput], language: str = "English", workers: int = 4,
                  progress: Callable[[int, int, int], None] | None = None) -> tuple[dict[int, dict], dict[int, Exception]]:
    """Nomi per tutti i cluster con `workers` richieste parallele.

    Ritorna (risultati, errori) per cluster: un cluster fallito non blocca gli altri.
    `progress(done, total, cluster_id)` viene chiamato dal thread chiamante (sicuro per Streamlit)."""
    start = time.perf_counter()
    results: dict[int, dict] = {}
    errors: dict[int, Exception] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="naming") as pool:
        futures = {pool.submit(name_cluster, llm_service, item, language): item.cluster_id for item in inputs}
        for done, fut in enumerate(as_completed(futures), start=1):
            cid = futures[fut]
            try:
                results[cid] = fut.result()
            except Exception as e:  # noqa: BLE001
                errors[cid] = e
                log.warning("Nome cluster %s fallito: %s", cid, e)
            if progress:
                progress(done, len(inputs), cid)
    log.info("Nomi generati per %d/%d cluster in %.1fs (%d richieste parallele)",
             len(results), len(inputs), time.perf_counter() - start, workers)
    return results, errors


def refine_names(llm_service, clusters: dict[str, dict], language: str = "English") -> dict[int, dict]:
    """Revisione globale: ritorna {cluster_id: {name, description}} solo per id esistenti."""
    rows = []
    for cid, meta in clusters.items():
        if cid == "-1" or not meta.get("name"):
            continue
        rows.append(f"- {cid} · {meta.get('size')} · {meta['name']} · {meta.get('description', '')} · "
                    f"{', '.join(meta.get('keywords', [])[:6])}")
    if len(rows) < 2:
        return {}
    messages = get_prompt("names_refinement").messages(language=language, clusters="\n".join(rows))
    out = llm_service.chat_json(messages, schema=SCHEMA_REFINE, temperature=0.2)
    valid = {int(c) for c in clusters if c != "-1"}
    refined = {}
    for item in out.get("clusters", []):
        try:
            cid = int(item["cluster_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if cid in valid and str(item.get("name", "")).strip():
            refined[cid] = {"name": item["name"].strip(), "description": str(item.get("description", "")).strip()}
    missing = valid - set(refined)
    if missing:
        log.warning("Revisione nomi: %d cluster non restituiti dall'LLM (restano invariati): %s",
                    len(missing), sorted(missing))
    return refined


def prompt_versions() -> dict[str, str]:
    return {pid: get_prompt(pid).version for pid in ("cluster_naming", "names_refinement")}


def preview(item: NamingInput, language: str = "English") -> str:
    """Testo completo che verrebbe inviato all'LLM (per la pagina Prompt)."""
    return json.dumps(naming_messages(item, language), indent=2, ensure_ascii=False)
