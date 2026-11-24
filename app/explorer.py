"""Componenti di visualizzazione di un clustering (risultato corrente o set salvato)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import common
from clustering import boundary_indices, central_indices, representative_indices

MAX_PLOT_POINTS = 6000


def cluster_label(cid: int, clusters: dict) -> str:
    if cid == -1:
        return "-1 · rumore"
    name = (clusters.get(str(cid)) or {}).get("name")
    return f"{cid} · {name}" if name else f"Cluster {cid}"


def metrics_row(metrics: dict, elapsed: float | None = None) -> None:
    items = [("n_clusters", "Cluster"), ("silhouette", "Silhouette ↑"), ("davies_bouldin", "Davies-Bouldin ↓"),
             ("noise_frac", "Rumore"), ("ambiguous_frac", "Punti ambigui")]
    shown = [(k, label) for k, label in items if metrics.get(k) is not None
             and not (k == "noise_frac" and metrics[k] == 0)]
    cols = st.columns(len(shown) + (1 if elapsed is not None else 0))
    for col, (k, label) in zip(cols, shown):
        v = metrics[k]
        val = f"{v:.1%}" if k in ("noise_frac", "ambiguous_frac") else common.fmt(v)
        col.metric(label, val, help=common.METRIC_HELP.get(k))
    if elapsed is not None:
        cols[-1].metric("Tempo", f"{elapsed:.1f} s")


def scatter(coords: np.ndarray | None, labels: np.ndarray, df: pd.DataFrame, clusters: dict,
            probs: np.ndarray | None = None, key: str = "scatter", cache: dict | None = None) -> None:
    """Mappa 2D dei cluster. `cache`: dizionario legato a questo risultato (coordinate ed etichette fisse)
    in cui riusare la figura finché i nomi dei cluster non cambiano: ricostruirla costa fino a ~1 s."""
    if coords is None:
        st.caption("Proiezione 2D non disponibile.")
        return
    ok = ~np.isnan(coords).any(axis=1)
    idx = np.where(ok)[0]
    if len(idx) > MAX_PLOT_POINTS:
        idx = np.random.default_rng(0).choice(idx, MAX_PLOT_POINTS, replace=False)
    names_key = tuple(sorted((c, (m or {}).get("name", "")) for c, m in clusters.items()))
    if cache is not None and cache.get("_fig_key") == names_key:
        fig = cache["_fig"]
    else:
        fig = _scatter_figure(coords, labels, df, clusters, probs, idx)
        if cache is not None:
            cache["_fig_key"], cache["_fig"] = names_key, fig
    st.plotly_chart(fig, width="stretch", key=key)
    if len(idx) < ok.sum():
        st.caption(f"Mostrati {len(idx):,} punti su {int(ok.sum()):,}.")


def _scatter_figure(coords, labels, df, clusters, probs, idx):
    plot = pd.DataFrame({
        "x": coords[idx, 0], "y": coords[idx, 1],
        "cluster": [cluster_label(int(l), clusters) for l in labels[idx]],
        "testo": df["Text"].iloc[idx].str.slice(0, 160).str.replace("\n", " ").to_numpy(),
        "Id": df["Id"].iloc[idx].to_numpy(),
    })
    if probs is not None:
        plot["confidenza"] = probs[idx].max(axis=1).round(3)
    order = sorted(plot["cluster"].unique(), key=lambda s: int(s.split(" ")[0]) if s.split(" ")[0].lstrip("-").isdigit() else 1e9)
    fig = px.scatter(plot, x="x", y="y", color="cluster", category_orders={"cluster": order},
                     hover_data={"x": False, "y": False, "Id": True, "testo": True,
                                 **({"confidenza": True} if probs is not None else {})},
                     opacity=0.75, height=560)
    fig.update_traces(marker={"size": 5})
    fig.update_layout(legend={"itemsizing": "constant"}, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def overview_table(labels: np.ndarray, clusters: dict, silhouettes: dict | None = None,
                   probs: np.ndarray | None = None) -> pd.DataFrame:
    ids, counts = np.unique(labels, return_counts=True)
    rows = []
    for cid, n in zip(ids, counts):
        meta = clusters.get(str(int(cid))) or {}
        row = {"id": int(cid), "nome": meta.get("name") or ("rumore" if cid == -1 else ""),
               "dimensione": int(n), "quota": n / len(labels)}
        if silhouettes is not None:
            row["silhouette"] = silhouettes.get(int(cid))
        if probs is not None and cid != -1:
            row["confidenza"] = float(probs[labels == cid].max(axis=1).mean())
        if meta.get("coherence"):
            row["coerenza"] = meta["coherence"]
        row["parole chiave"] = ", ".join(meta.get("keywords", [])[:8])
        row["descrizione"] = meta.get("description", "")
        rows.append(row)
    return pd.DataFrame(rows)


def show_overview(table: pd.DataFrame) -> None:
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "quota": st.column_config.ProgressColumn("quota", format="percent", min_value=0, max_value=1),
            "silhouette": st.column_config.NumberColumn(format="%.3f"),
            "confidenza": st.column_config.NumberColumn(format="%.2f"),
            "coerenza": st.column_config.NumberColumn(help="Omogeneità 1-5 stimata dall'LLM", format="%d/5"),
        },
    )


def cluster_detail(cid: int, X: np.ndarray | None, labels: np.ndarray, probs: np.ndarray | None,
                   df: pd.DataFrame, clusters: dict) -> None:
    meta = clusters.get(str(cid)) or {}
    if meta.get("description"):
        st.markdown(f"**{meta.get('name', '')}** — {meta['description']}")
    if meta.get("evidence"):
        st.caption("Evidenze: " + " · ".join(meta["evidence"]))
    if meta.get("keywords"):
        st.caption("Parole chiave: " + ", ".join(meta["keywords"]))

    def show(idx, title, with_conf=False):
        if len(idx) == 0:
            return
        st.markdown(f"**{title}**")
        sub = df.iloc[idx][["Id", "Text"] + [c for c in ("Summary", "Score") if c in df.columns]].copy()
        if with_conf and probs is not None:
            sub.insert(1, "confidenza", probs[idx].max(axis=1).round(3))
            if probs.shape[1] > 1:
                second = np.argsort(probs[idx], axis=1)[:, -2]
                sub.insert(2, "2° cluster", [cluster_label(int(c), clusters) for c in second])
        st.dataframe(sub, hide_index=True, width="stretch",
                     column_config={"Text": st.column_config.TextColumn(width="large")})

    if X is not None:
        show(central_indices(X, labels, cid, n=5), "Più centrali (prototipi)")
        show(representative_indices(X, labels, cid, per_tercile=2, seed=7), "Rappresentativi (vicini → periferici)")
    else:
        idx = np.where(labels == cid)[0][:8]
        show(idx, "Esempi")
    if cid != -1:
        show(boundary_indices(probs, labels, cid, n=5), "Casi di confine (confidenza minima)", with_conf=True)
