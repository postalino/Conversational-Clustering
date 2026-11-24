import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import common
import explorer
from clustering import ALGORITHMS, PreprocessConfig, cluster_keywords, default_hdbscan_params, k_sweep, preprocess, run_clustering
from labeling import build_inputs, keywords_by_cluster, name_clusters, prompt_versions, refine_names
from projection import METHODS as PROJ_METHODS, project_2d, umap_ready
from runs import Run, new_run_id, save_run, set_selected

st.title("🔬 Clustering iniziale")

ss = st.session_state
if not common.workspace_ready():
    st.warning("Servono un campione e i relativi embedding: vedi **Dataset** ed **Embedding**.", icon="⚠️")
    st.stop()
if ss.X_model != common.emb_cfg().model:
    st.warning(f"Gli embedding caricati sono di `{ss.X_model}`, ma il modello selezionato è "
               f"`{common.emb_cfg().model}`. Il clustering userà `{ss.X_model}`.", icon="⚠️")

df: pd.DataFrame = ss.sample_df
X: np.ndarray = ss.X
n = len(X)
st.caption(f"Campione: **{n:,}** testi · embedding `{ss.X_model}` ({X.shape[1]} dimensioni)")

def preprocessed(pre: PreprocessConfig) -> np.ndarray:
    """Preprocessing in cache di sessione: cambiando solo l'algoritmo non si ricalcola la PCA."""
    ids_hash = hashlib.md5(df["Id"].to_numpy().tobytes()).hexdigest()
    key = (ss.X_model, ids_hash, tuple(asdict(pre).items()))
    cache = ss.setdefault("prep_cache", {})
    if key not in cache:
        with st.spinner("Preprocessing (normalizzazione / riduzione)…"):
            cache.clear()
            cache[key] = preprocess(X, pre)
    return cache[key]


tab_run, tab_k, tab_res = st.tabs(["⚙️ Configura ed esegui", "📈 Scelta di k", "🔍 Risultato"])

# ============================================================================= configurazione
with tab_run:
    with st.container(border=True):
        st.subheader("Preprocessing")
        c1, c2, c3 = st.columns(3)
        l2 = c1.checkbox("Normalizzazione L2", value=True,
                         help="Con vettori normalizzati la distanza euclidea equivale alla distanza coseno")
        reduction = c2.selectbox("Riduzione dimensionale", ["pca", "umap", "none"],
                                 format_func={"none": "Nessuna", "pca": "PCA", "umap": "UMAP"}.get,
                                 help="PCA: veloce, consigliata per K-Means. UMAP: preserva meglio la "
                                      "struttura locale, spesso migliore per HDBSCAN.")
        n_comp = c3.slider("Componenti", 2, min(256, X.shape[1]), 50 if reduction != "umap" else 10,
                           disabled=reduction == "none")
        pre = PreprocessConfig(l2_normalize=l2, reduction=reduction, n_components=n_comp)

    with st.container(border=True):
        st.subheader("Algoritmo")
        algo = st.radio("Algoritmo", list(ALGORITHMS), format_func=ALGORITHMS.get, horizontal=True,
                        label_visibility="collapsed")
        params: dict = {}
        c1, c2, c3, c4 = st.columns(4)
        params["seed"] = c4.number_input("Seed", 0, 10_000, 42)

        if algo == "kmeans":
            params["k"] = c1.number_input("Numero di cluster k", 2, min(200, n - 1), min(8, n - 1))
            params["n_init"] = c2.number_input("Inizializzazioni (n_init)", 1, 50, 10)
            params["minibatch"] = c3.checkbox("MiniBatch K-Means", value=n > 20_000,
                                              help="Più veloce su grandi campioni, leggermente meno preciso")
            params["soft_temperature"] = st.slider(
                "Temperatura assegnazioni soft", 0.005, 0.5, 0.05, 0.005,
                help="Probabilità soft = softmax sulle distanze dai centroidi: temperatura bassa = assegnazioni nette")
        elif algo == "hdbscan":
            d = default_hdbscan_params(n)
            params["min_cluster_size"] = c1.number_input("min_cluster_size", 2, max(3, n // 2), d["min_cluster_size"],
                                                         help="Dimensione minima di un cluster")
            params["min_samples"] = c2.number_input("min_samples", 1, max(2, n // 2), d["min_samples"],
                                                    help="Più alto = più conservativo (più punti come rumore)")
            params["cluster_selection_method"] = c3.selectbox("Selezione cluster", ["eom", "leaf"],
                                                              help="'leaf' produce cluster più piccoli e omogenei")

    with st.container(border=True):
        st.subheader("Visualizzazione")
        c1, c2 = st.columns(2)
        proj = c1.selectbox("Proiezione 2D", list(PROJ_METHODS), format_func=PROJ_METHODS.get)
        proj_max = c2.number_input("Max punti proiettati", 500, 50_000, 10_000, step=1000, disabled=proj == "pca")
        if proj == "umap" and not umap_ready():
            st.caption("⏳ UMAP è ancora in compilazione in background (solo al primo avvio): se lanci ora attenderà.")

    if st.button("▶ Esegui clustering", type="primary", width="stretch"):
        with common.guarded("Clustering", algorithm=algo, params=params, preprocess=asdict(pre)):
            with st.status("Clustering in corso…", expanded=True) as status:
                Xp = preprocessed(pre)
                st.write(f"Preprocessing: {Xp.shape[0]:,} × {Xp.shape[1]}")
                res = run_clustering(Xp, algo, params)
                st.write(f"{ALGORITHMS[algo]}: {res.metrics['n_clusters']} cluster in {res.elapsed_s:.1f}s")

                # Proiezione 2D e parole chiave sono indipendenti: in parallelo.
                t0 = time.perf_counter()
                proj_idx = np.arange(n)
                if proj != "pca" and n > proj_max:
                    proj_idx = np.random.default_rng(0).choice(n, int(proj_max), replace=False)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_proj = pool.submit(project_2d, Xp[proj_idx], proj)
                    f_kw = pool.submit(cluster_keywords, df["Text"].tolist(), res.labels)
                    coords = np.full((n, 2), np.nan, dtype=np.float32)
                    coords[proj_idx] = f_proj.result()
                    kws = f_kw.result()
                st.write(f"Proiezione {PROJ_METHODS[proj]} e parole chiave in {time.perf_counter() - t0:.1f}s")

                sizes = dict(zip(*np.unique(res.labels, return_counts=True)))
                clusters = {str(c): {"name": "", "description": "", "evidence": [], "coherence": None,
                                     "keywords": kws.get(c, []), "size": int(sizes.get(c, 0))} for c in res.cluster_ids}
                if -1 in sizes:
                    clusters["-1"] = {"name": "Rumore / outlier", "description": "Punti non assegnati da HDBSCAN.",
                                      "evidence": [], "keywords": [], "size": int(sizes[-1])}
                ss.result = {"res": res, "pre": pre, "Xp": Xp, "coords": coords, "clusters": clusters, "labels_model": None, "prompts": None, "saved_as": None}
                status.update(label="Clustering completato — vedi la scheda *Risultato*", state="complete")

# ============================================================================= scelta di k
with tab_k:
    st.caption("Esegue K-Means per più valori di k (in parallelo) e mostra inerzia (metodo del gomito) e "
               "silhouette, come supporto alla scelta: non sostituisce il giudizio umano. "
               "HDBSCAN non ne ha bisogno: il numero di cluster lo trova da solo.")
    c1, c2, c3 = st.columns(3)
    k_min = c1.number_input("k minimo", 2, 199, 2)
    k_max = c2.number_input("k massimo", 3, 200, 20)
    k_step = c3.number_input("passo", 1, 20, 1)
    ks = [k for k in range(int(k_min), int(k_max) + 1, int(k_step)) if k < n]
    if k_min > k_max:
        st.warning("Il k minimo deve essere ≤ del k massimo.", icon="⚠️")
    if st.button("Calcola curva", type="primary", disabled=not ks):
        bar = st.progress(0.0)
        with common.guarded("Scelta di k", ks=ks):
            t0 = time.perf_counter()
            rows = k_sweep(preprocessed(pre), ks,
                           progress=lambda i, t: bar.progress(i / t, text=f"{i}/{t} valori di k"))
            ss.k_sweep = (pd.DataFrame(rows), time.perf_counter() - t0)
    table, secs = ss.get("k_sweep") or (None, 0.0)
    if table is not None and not table.empty:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Inerzia ↓ (cerca il gomito)", "Silhouette ↑"))
        for i, col in enumerate(["inerzia", "silhouette"], start=1):
            fig.add_trace(go.Scatter(x=table["k"], y=table[col], mode="lines+markers", name=col), row=1, col=i)
        fig.update_layout(height=320, showlegend=False, margin={"t": 40, "b": 10})
        st.plotly_chart(fig, width="stretch")
        sil = pd.to_numeric(table["silhouette"], errors="coerce")
        if sil.notna().any():
            best = table.loc[sil.idxmax()]
            st.info(f"Silhouette massima con k = **{int(best['k'])}** ({best['silhouette']:.3f}) — "
                    f"{len(table)} valori in {secs:.1f}s", icon="📌")

# ============================================================================= risultato
with tab_res:
    r = ss.result
    if r is None:
        st.info("Esegui un clustering nella prima scheda.", icon="ℹ️")
        st.stop()
    res = r["res"]
    clusters = r["clusters"]
    st.markdown(f"**{ALGORITHMS[res.algorithm]}** · `{res.params}`")
    explorer.metrics_row(res.metrics, res.elapsed_s)

    # --------------------------------------------------- nomi con LLM
    with st.container(border=True):
        st.markdown("**✨ Nomi e descrizioni con LLM**")
        c1, c2, c3 = st.columns([1, 1, 2])
        lang = c1.selectbox("Lingua dei nomi", list(common.LANGUAGES), format_func=common.LANGUAGES.get,
                            index=common.language_index(ss.label_language))
        ss.label_language = lang
        only_missing = c2.checkbox("Solo cluster senza nome", value=False)
        refine = c3.checkbox("Revisione globale finale", value=True,
                             help="Una chiamata in più che vede tutti i nomi insieme e li rende distinti e coerenti")
        chat = common.chat_cfg()
        workers = chat.effective_workers(ss.llm_workers)  # Ollama serve una richiesta alla volta
        st.caption(f"Modello `{chat.model}` · {workers} richieste parallele · "
                   f"prompt modificabili nella pagina **Prompt**")
        if st.button("Genera nomi", type="primary"):
            llm = common.llm_service(chat)
            ids = [c for c in res.cluster_ids if not (only_missing and clusters[str(c)].get("name"))]
            inputs = build_inputs(r["Xp"], res.labels, df["Text"].tolist(), ids, keywords_by_cluster(clusters))
            bar = st.progress(0.0, text="Generazione nomi…")
            t0 = time.perf_counter()
            results, errors = name_clusters(
                llm, inputs, language=lang, workers=workers,
                progress=lambda d, t, cid: bar.progress(d / t, text=f"{d}/{t} cluster (ultimo: {cid})"))
            for cid, out in results.items():
                clusters[str(cid)].update(out)
            for cid, err in errors.items():
                with common.guarded(f"Nome cluster {cid}", model=llm.model):
                    raise err
            if refine and len(results) > 1:
                with common.guarded("Revisione globale dei nomi", model=llm.model):
                    with st.spinner("Revisione globale…"):
                        for cid, upd in refine_names(llm, clusters, language=lang).items():
                            clusters[str(cid)].update(upd)
            r["labels_model"], r["prompts"] = llm.model, prompt_versions()
            bar.progress(1.0, text=f"{len(results)}/{len(inputs)} cluster in {time.perf_counter() - t0:.1f}s")

    explorer.scatter(r["coords"], res.labels, df, clusters, res.probs, key="res_scatter", cache=r)
    explorer.show_overview(explorer.overview_table(res.labels, clusters, res.cluster_silhouette, res.probs))

    # --------------------------------------------------- dettaglio + modifica manuale
    st.subheader("Dettaglio cluster")
    ids_all = sorted({int(c) for c in clusters})
    cid = st.selectbox("Cluster", ids_all, format_func=lambda c: explorer.cluster_label(c, clusters))
    meta = clusters[str(cid)]
    with st.form(f"edit_{cid}"):
        e1, e2 = st.columns([1, 2])
        new_name = e1.text_input("Nome", value=meta.get("name", ""))
        new_desc = e2.text_input("Descrizione", value=meta.get("description", ""))
        if st.form_submit_button("Salva modifica"):
            meta.update(name=new_name, description=new_desc)
            st.rerun()
    explorer.cluster_detail(cid, r["Xp"], res.labels, res.probs, df, clusters)

    # --------------------------------------------------- salvataggio
    st.divider()
    with st.container(border=True):
        st.subheader("💾 Salva come set iniziale")
        if r.get("saved_as"):
            st.success(f"Già salvato come `{r['saved_as']}`. Salvando di nuovo si crea un nuovo set.")
        default_name = f"{ALGORITHMS[res.algorithm]} · {res.metrics['n_clusters']} cluster · {datetime.now():%d/%m %H:%M}"
        s1, s2 = st.columns([2, 1])
        run_name = s1.text_input("Nome del set", value=default_name)
        select_now = s2.checkbox("Usalo come set iniziale per l'agente", value=False)
        notes = st.text_area("Note", placeholder="Perché questo set? Cosa ti convince / non ti convince?")
        if st.button("Salva set", type="primary"):
            with common.guarded("Salvataggio set iniziale"):
                run = Run(
                    id=new_run_id(res.algorithm), name=run_name, created=datetime.now().isoformat(timespec="seconds"),
                    dataset=common.dataset_slug(), embedding_model=ss.X_model,
                    sample=ss.sample_cfg.to_dict() | {"n": n}, preprocess=asdict(r["pre"]),
                    algorithm=res.algorithm, params=res.params, metrics=res.metrics | {"elapsed_s": res.elapsed_s},
                    clusters=clusters, notes=notes, labels_model=r.get("labels_model"),
                    prompt_versions=r.get("prompts"),
                    ids=df["Id"].to_numpy(), labels=res.labels, probs=res.probs, coords=r["coords"],
                )
                save_run(run)
                if select_now:
                    set_selected(run.id)
                r["saved_as"] = run.id
                st.success(f"Salvato `{run.id}`" + (" e scelto come set iniziale ✅" if select_now else ""))
