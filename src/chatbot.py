"""Chatbot semplice (placeholder della futura parte agentica): nessuna azione sui cluster."""

from __future__ import annotations

from prompts import get_prompt
from runs import Run

MAX_HISTORY_MESSAGES = 20  # ultimi messaggi inviati al modello: limita token e costo per turno


def run_context(run: Run, max_clusters: int = 60) -> str:
    m = run.metrics
    lines = [
        "## Current initial clustering",
        f"Name: {run.name} · algorithm: {run.algorithm} · dataset: {run.dataset} · "
        f"texts: {run.sample.get('n')} · embedding model: {run.embedding_model}",
        f"Quality: silhouette {_f(m.get('silhouette'))}, Davies-Bouldin {_f(m.get('davies_bouldin'))}, "
        f"noise {_pct(m.get('noise_frac'))}, ambiguous points {_pct(m.get('ambiguous_frac'))}",
        "",
        "## Clusters (id · size · name · coherence 1-5 · description · keywords)",
    ]
    items = sorted(run.clusters.items(), key=lambda kv: -int(kv[1].get("size") or 0))
    for cid, c in items[:max_clusters]:
        lines.append(f"- {cid} · {c.get('size')} · {c.get('name') or 'unnamed'} · {c.get('coherence') or '?'} · "
                     f"{c.get('description') or ''} · {', '.join(c.get('keywords', [])[:6])}")
    if len(items) > max_clusters:
        lines.append(f"- … other {len(items) - max_clusters} smaller clusters omitted")
    return "\n".join(lines)


def build_messages(history: list[dict], run: Run | None) -> list[dict]:
    context = run_context(run) if run is not None else "(No initial clustering selected yet.)"
    system, _ = get_prompt("chatbot_system").render(context=context)
    turns = [{"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")]
    return [{"role": "system", "content": system}] + turns[-MAX_HISTORY_MESSAGES:]


def _f(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def _pct(v) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "n/a"
