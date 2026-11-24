import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import common
import explorer
from clustering import ALGORITHMS
from dataset import SampleConfig
from runs import agreement, delete_run, get_selected, list_runs, set_selected, update_meta

st.title("📌 Set iniziali")
st.caption("Tutti i clustering salvati. Confrontali e scegli quello da cui partirà l'agente.")

runs = list_runs()
if not runs:
    st.info("Nessun set salvato: eseguine uno nella pagina **Clustering** e salvalo.", icon="ℹ️")
    st.stop()

by_id = {r.id: r for r in runs}
selected = get_selected()
current_dataset = common.dataset_slug()

# ----------------------------------------------------------------------------- set selezionato
if selected and selected not in by_id:
    st.error(f"Il set scelto `{selected}` esiste ma non è leggibile (vedi Log & Errori). "
             "Scegline un altro o deselezionalo.", icon="🚨")
    if st.button("Deseleziona il set illeggibile"):
        set_selected(None)
        st.rerun()
    selected = None
if selected:
    sel = by_id[selected]
    n_txt = sel.sample.get("n")
    n_label = f"{n_txt:,}" if isinstance(n_txt, int) else "?"
    st.success(f"**Set iniziale scelto:** {sel.name}  \n`{sel.id}` · {ALGORITHMS.get(sel.algorithm, sel.algorithm)} · "
               f"{sel.metrics.get('n_clusters')} cluster · {n_label} testi · `{sel.embedding_model}`", icon="✅")
else:
    st.warning("Nessun set iniziale scelto per l'agente.", icon="⚠️")

# ----------------------------------------------------------------------------- tabella
rows = []
for r in runs:
    m = r.metrics
    named = sum(1 for c, v in r.clusters.items() if c != "-1" and v.get("name"))
    rows.append({
        "scelto": r.id == selected, "nome": r.name, "id": r.id, "algoritmo": ALGORITHMS.get(r.algorithm, r.algorithm),
        "cluster": m.get("n_clusters"), "testi": r.sample.get("n"), "silhouette": m.get("silhouette"),
        "davies_bouldin": m.get("davies_bouldin"), "rumore": m.get("noise_frac"),
        "ambigui": m.get("ambiguous_frac"), "nomi": f"{named}/{m.get('n_clusters')}",
        "embedding": r.embedding_model, "creato": r.created.replace("T", " "),
    })
table = pd.DataFrame(rows)
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "scelto": st.column_config.CheckboxColumn("✓", width="small"),
    "silhouette": st.column_config.NumberColumn(format="%.3f"),
    "davies_bouldin": st.column_config.NumberColumn("DB", format="%.3f"),
    "rumore": st.column_config.NumberColumn(format="percent"),
    "ambigui": st.column_config.NumberColumn(format="percent"),
})

tab_detail, tab_cmp = st.tabs(["🔍 Dettaglio e scelta", "⚖️ Confronto"])

# ----------------------------------------------------------------------------- dettaglio
with tab_detail:
    run_id = st.selectbox("Set", [r.id for r in runs], format_func=lambda i: f"{by_id[i].name}  ({i})",
                          index=[r.id for r in runs].index(selected) if selected else 0)
    run = common.load_run_cached(run_id)
    same_dataset = run.dataset == current_dataset
    if not same_dataset:
        st.warning(f"Questo set è stato creato sul dataset `{run.dataset}`, ma il dataset attuale è "
                   f"`{current_dataset}`: testi ed embedding non sono disponibili (gli id non corrispondono). "
                   "Riscarica quel dataset per esplorarlo o aprirlo.", icon="⚠️")

    b1, b2, b3, b4 = st.columns(4)
    if run_id != selected:
        if b1.button("✅ Usa come set iniziale", type="primary", width="stretch"):
            set_selected(run_id)
            st.rerun()
    elif b1.button("Deseleziona", width="stretch"):
        set_selected(None)
        st.rerun()
    if b2.button("Apri nel workspace", width="stretch", disabled=not same_dataset,
                 help="Carica campione ed embedding di questo set, per ripartire da qui nella pagina Clustering"):
        with common.guarded("Apertura set nel workspace", run=run_id):
            ss = st.session_state
            # Tutto viene preparato in variabili locali: lo stato della sessione cambia solo se
            # testi ed embedding sono entrambi disponibili (niente workspace "misto" in caso di errore).
            new_cfg = SampleConfig.from_dict(run.sample)
            new_df = common.sample_from_ids(run.ids, run.sample)
            store = common.current_store(text_col=new_cfg.text_col, model=run.embedding_model)
            missing = store.missing_ids(new_df["Id"])
            if missing:
                raise RuntimeError(f"{len(missing):,} testi del set non hanno embedding in cache per "
                                   f"`{run.embedding_model}`: calcolali nella pagina Embedding")
            new_X = store.get(new_df["Id"].to_numpy())

            if run.embedding_model != common.emb_cfg().model:
                st.info(f"Modello embedding impostato su `{run.embedding_model}` per coerenza con il set.")
                ss.emb_cfg = ss.emb_cfg.with_model(run.embedding_model)
            common.set_workspace(new_cfg, new_df, new_X, run.embedding_model)
            st.success("Workspace caricato: vai su **Clustering**.")
    with b3.popover("Rinomina / note", width="stretch"):
        new_name = st.text_input("Nome", value=run.name)
        new_notes = st.text_area("Note", value=run.notes)
        if st.button("Salva"):
            run.name, run.notes = new_name, new_notes
            update_meta(run)
            st.rerun()
    with b4.popover("🗑️ Elimina", width="stretch"):
        st.write(f"Eliminare definitivamente `{run_id}`?")
        if st.button("Conferma eliminazione", type="primary"):
            with common.guarded("Eliminazione set", run=run_id):
                delete_run(run_id)
                st.rerun()

    explorer.metrics_row(run.metrics, run.metrics.get("elapsed_s"))
    if run.notes:
        st.caption(f"📝 {run.notes}")

    df = None
    if same_dataset:
        with common.guarded("Caricamento testi del set", run=run_id):
            df = common.sample_from_ids(run.ids, run.sample)
    if df is not None:
        fig_cache = st.session_state.setdefault("_run_figs", {}).setdefault(run_id, {})
        explorer.scatter(run.coords, run.labels, df, run.clusters, run.probs, key="run_scatter", cache=fig_cache)
        explorer.show_overview(explorer.overview_table(run.labels, run.clusters, probs=run.probs))
        cid = st.selectbox("Cluster", sorted({int(c) for c in run.clusters}),
                           format_func=lambda c: explorer.cluster_label(c, run.clusters))
        explorer.cluster_detail(cid, None, run.labels, run.probs, df, run.clusters)

    with st.expander("Configurazione completa (JSON)"):
        st.json(run.meta())
    cluster_names = {int(c): (meta or {}).get("name", "") for c, meta in run.clusters.items()}
    st.download_button(
        "⬇️ Esporta assegnazioni (CSV)",
        # callable: il CSV viene generato solo al clic, non a ogni rerun
        lambda: pd.DataFrame({"Id": run.ids, "cluster_id": run.labels,
                              "cluster_name": pd.Series(run.labels).map(cluster_names).fillna("").to_numpy()})
        .to_csv(index=False).encode("utf-8"),
        file_name=f"{run.id}_assignments.csv", mime="text/csv",
    )

# ----------------------------------------------------------------------------- confronto
with tab_cmp:
    chosen = st.multiselect("Set da confrontare", [r.id for r in runs], default=[r.id for r in runs[:min(4, len(runs))]],
                            format_func=lambda i: by_id[i].name)
    if len(chosen) >= 2:
        loaded = {i: common.load_run_cached(i) for i in chosen}
        ari = np.full((len(chosen), len(chosen)), np.nan)
        for a, ia in enumerate(chosen):
            for b in range(a, len(chosen)):  # l'ARI è simmetrico: metà delle coppie basta
                ag = agreement(loaded[ia], loaded[chosen[b]])
                if ag:
                    ari[a, b] = ari[b, a] = ag["ari"]
        names = [by_id[i].name[:40] for i in chosen]
        st.markdown("**Accordo tra set (Adjusted Rand Index sugli elementi in comune)** — 1 = partizioni identiche, "
                    "~0 = accordo casuale")
        fig = px.imshow(ari, x=names, y=names, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale="Blues")
        fig.update_layout(height=120 + 60 * len(chosen), margin={"t": 10, "b": 10})
        st.plotly_chart(fig, width="stretch")

        metric_cols = ["n_clusters", "silhouette", "davies_bouldin", "noise_frac", "ambiguous_frac"]
        mt = pd.DataFrame([{"set": by_id[i].name[:40], **{k: by_id[i].metrics.get(k) for k in metric_cols}}
                           for i in chosen])
        st.dataframe(mt, hide_index=True, width="stretch")

        st.markdown("**Distribuzione delle dimensioni dei cluster**")
        sizes = pd.concat([
            pd.DataFrame({"set": by_id[i].name[:40],
                          "dimensione": np.unique(loaded[i].labels[loaded[i].labels != -1], return_counts=True)[1]})
            for i in chosen
        ])
        st.plotly_chart(px.box(sizes, x="set", y="dimensione", points="all", height=350), width="stretch")
    else:
        st.caption("Seleziona almeno due set.")
