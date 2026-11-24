"""Registro centralizzato dei prompt.

Ogni prompt ha un testo di default (qui) e può essere sovrascritto dalla GUI (pagina "Prompt"):
le modifiche sono salvate in `data/prompts.json`. I template usano `string.Template`
($variabile), così le parentesi graffe del JSON negli esempi non creano problemi.
Ogni set salvato registra la versione (hash) dei prompt usati, per riproducibilità.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from string import Template

from config import DATA_DIR
from logging_setup import get_logger

log = get_logger(__name__)

OVERRIDES_PATH = DATA_DIR / "prompts.json"


@dataclass(frozen=True)
class PromptSpec:
    id: str
    title: str
    purpose: str
    system: str
    user: str
    variables: tuple[str, ...]


DEFAULTS: dict[str, PromptSpec] = {}


def _register(spec: PromptSpec) -> None:
    DEFAULTS[spec.id] = spec


# ----------------------------------------------------------------------------- nomi dei cluster
_register(PromptSpec(
    id="cluster_naming",
    title="Nome e descrizione di un cluster",
    purpose="Chiamato una volta per cluster, in parallelo. Riceve parole chiave distintive, parole chiave dei "
            "cluster più simili (per nomi contrastivi), esempi tipici e un campione dal centro alla periferia.",
    system="""You label clusters of short texts (for example product reviews) produced by an embedding-based clustering algorithm.

Rules:
- name: 2-5 words, a specific noun phrase for what the texts have in common (topic, product type, issue, sentiment if it is the defining trait). Avoid empty words such as "Reviews", "Feedback", "Various", "General", "Mixed".
- description: one or two neutral sentences on what the texts share and, when useful, what separates them from the similar clusters listed.
- evidence: up to 3 short recurring elements you actually observe in the examples. Do not invent.
- coherence: integer 1-5. 5 = every example clearly shares one theme; 3 = a dominant theme with notable exceptions; 1 = no common theme.
- Rely only on the examples and keywords provided. If the cluster is heterogeneous, say it in the description and lower the coherence instead of forcing a specific name.
- Write name, description and evidence in $language.

Return only JSON that matches the schema.""",
    user="""Cluster with $size texts ($share of the sample).

Distinctive keywords (c-TF-IDF): $keywords

Keywords of the most similar clusters (the name must distinguish this cluster from them):
$neighbors

Most typical texts (closest to the cluster centre):
$central

Texts sampled across the cluster (from central to peripheral):
$diverse

Write name, description and evidence in $language, even if the texts are in another language.""",
    variables=("language", "size", "share", "keywords", "neighbors", "central", "diverse"),
))

_register(PromptSpec(
    id="names_refinement",
    title="Revisione globale dei nomi",
    purpose="Una sola chiamata dopo i nomi per cluster: vede tutti i cluster insieme e rende i nomi distinti "
            "e coerenti nello stile, senza cambiare la partizione.",
    system="""You review a complete set of cluster labels that were written independently, to make them work well together.

- Every name must be distinct. When two names overlap, make the difference explicit.
- Use a consistent style and granularity (similar length, same kind of noun phrase).
- Keep names that are already good; change only what improves clarity or distinctiveness.
- Keep descriptions short and consistent with the new names.
- Do not merge, split, add or remove clusters: return exactly the same cluster ids.
- Write in $language.

Return only JSON that matches the schema.""",
    user="""Clusters (id · size · current name · description · keywords):
$clusters

Write every name and description in $language (translate any that are in another language).""",
    variables=("language", "clusters"),
))

# ----------------------------------------------------------------------------- chatbot
_register(PromptSpec(
    id="chatbot_system",
    title="Chatbot (system prompt)",
    purpose="System prompt del chatbot. $context contiene il riassunto del set iniziale scelto (se attivo).",
    system="""You are the assistant of an interactive text-clustering tool. The user is exploring an initial clustering of a text dataset and will later refine it with an agent.

- Answer in the user's language, concisely. Use short bullet lists for comparisons.
- Ground your answers in the clustering context below. If something is not in the context, say so instead of guessing.
- You cannot modify the clustering yet. If the user asks to merge, split, rename or move items, explain what the change would mean and whether the context supports it, and say that agent actions are not active yet.

$context""",
    user="",
    variables=("context",),
))


# ----------------------------------------------------------------------------- API
@dataclass
class Prompt:
    id: str
    system: str
    user: str
    customized: bool

    @property
    def version(self) -> str:
        return hashlib.sha1((self.system + "\x00" + self.user).encode("utf-8")).hexdigest()[:8]

    def render(self, **values) -> tuple[str, str]:
        """(system, user) con le variabili sostituite. Variabili mancanti restano visibili ($nome)."""
        vals = {k: str(v) for k, v in values.items()}
        return Template(self.system).safe_substitute(vals), Template(self.user).safe_substitute(vals)

    def messages(self, **values) -> list[dict[str, str]]:
        """Messaggi chat (system + user) pronti per l'LLM."""
        system, user = self.render(**values)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _load_overrides() -> dict:
    if OVERRIDES_PATH.exists():
        try:
            return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.error("data/prompts.json non valido, uso i default: %s", e)
    return {}


def get_prompt(prompt_id: str) -> Prompt:
    spec = DEFAULTS[prompt_id]
    o = _load_overrides().get(prompt_id, {})
    system, user = o.get("system", spec.system), o.get("user", spec.user)
    return Prompt(prompt_id, system, user, customized=(system != spec.system or user != spec.user))


def save_prompt(prompt_id: str, system: str, user: str) -> None:
    data = _load_overrides()
    spec = DEFAULTS[prompt_id]
    if system == spec.system and user == spec.user:
        data.pop(prompt_id, None)
    else:
        data[prompt_id] = {"system": system, "user": user}
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Prompt '%s' salvato (versione %s)", prompt_id, get_prompt(prompt_id).version)


def reset_prompt(prompt_id: str) -> None:
    spec = DEFAULTS[prompt_id]
    save_prompt(prompt_id, spec.system, spec.user)


def missing_variables(prompt_id: str, system: str, user: str) -> list[str]:
    """Variabili previste dal codice che il testo modificato non usa più (avviso nella GUI)."""
    text = system + user
    return [v for v in DEFAULTS[prompt_id].variables if f"${v}" not in text and f"${{{v}}}" not in text]
