import time

import pandas as pd
import streamlit as st

import common
from config import DATA_DIR
from embeddings import EmbeddingService, compute_missing, estimate_tokens, import_legacy_cache, list_stores

st.title("🧬 Embedding")

ss = st.session_state
cfg = common.emb_cfg()
slug = common.dataset_slug()
store = common.current_store()

c1, c2, c3 = st.columns(3)
c1.metric("Modello", cfg.model)
c2.metric("Provider", cfg.provider)
c3.metric("Vettori in cache (questo modello)", f"{store.count():,}")
st.caption("Il modello si cambia in **Modelli & API**. La cache è separata per dataset e modello.")

# ----------------------------------------------------------------------------- calcolo
df = ss.sample_df
with st.container(border=True):
    st.subheader("Embedding del campione")
    if df is None:
        st.info("Crea prima un campione nella pagina **Dataset**.", icon="ℹ️")
    else:
        missing = store.missing_ids(df["Id"])
        todo = df[df["Id"].isin(set(missing))]
        c = st.columns(4)
        c[0].metric("Testi nel campione", f"{len(df):,}")
        c[1].metric("Già in cache", f"{len(df) - len(missing):,}")
        c[2].metric("Da calcolare", f"{len(missing):,}")
        tokens = estimate_tokens(todo["Text"]) if len(todo) else 0
        c[3].metric("Token stimati", f"{tokens:,}")

        if missing:
            b1, b2 = st.columns(2)
            if cfg.serial:
                st.info("Ollama elabora una richiesta alla volta (su CPU circa 1 testo/s con modelli come bge-m3): "
                        "valori consigliati 1 richiesta e 16 testi per richiesta.", icon="🖥️")
            batch = b1.select_slider("Testi per richiesta", [8, 16, 32, 64, 100, 128, 256, 512],
                                     value=cfg.default_embed_batch,
                                     help="Ridurlo se compaiono errori di dimensione della richiesta o timeout.")
            workers = b2.slider("Richieste parallele", 1, 16, cfg.effective_workers(ss.emb_workers),
                                help="Più richieste contemporanee = più veloce, finché il provider non applica "
                                     "limiti (errore 429). Per Ollama locale conviene 1-2.")
            if not cfg.serial:
                ss.emb_workers = workers
            if st.button(f"Calcola {len(missing):,} embedding", type="primary"):
                bar = st.progress(0.0, text="Avvio…")
                t0 = time.perf_counter()
                with common.guarded("Calcolo embedding", model=cfg.model, n=len(missing), batch=batch, workers=workers):
                    done = compute_missing(
                        EmbeddingService(cfg), store, todo, batch_size=batch, workers=workers,
                        progress=lambda d, t: bar.progress(d / t, text=f"{d:,}/{t:,} testi"),
                    )
                    secs = time.perf_counter() - t0
                    bar.progress(1.0, text=f"Completato: {done:,} testi in {secs:.0f}s ({done / max(secs, 1e-9):.1f} testi/s)")
                    if common.load_embeddings_for_sample():  # workspace pronto senza passaggi in più
                        ss.result = None  # (il messaggio "Embedding pronti" compare qui sotto)
                st.caption("Gli embedding calcolati prima di un eventuale errore sono salvati: "
                           "rilanciando si riparte da dove ci si era fermati.")
        if not store.missing_ids(df["Id"]):
            if common.workspace_ready() and ss.X_model == cfg.model:
                st.success(f"Embedding pronti: matrice {ss.X.shape[0]:,} × {ss.X.shape[1]}", icon="✅")
            elif st.button("Carica embedding nel workspace", type="primary"):
                with common.guarded("Caricamento embedding"):
                    common.load_embeddings_for_sample()
                    ss.result = None
                    st.rerun()

# ----------------------------------------------------------------------------- cache
with st.container(border=True):
    st.subheader("Cache su disco")
    stores = list_stores(slug)
    if stores:
        rows = [s.info() for s in stores]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with st.popover("Elimina una cache"):
            target = st.selectbox("Cache", [s.model for s in stores])
            if st.button("Elimina definitivamente", type="primary"):
                with common.guarded("Eliminazione cache", model=target):
                    next(s for s in stores if s.model == target).delete()
                    st.rerun()
    else:
        st.caption("Nessuna cache per questo dataset.")

    legacy = DATA_DIR / "embeddings.npz"
    if legacy.exists():
        st.markdown("**Cache del vecchio formato trovata** (`data/embeddings.npz`, indicizzata solo per Id). "
                    "Importala indicando il modello con cui era stata generata.")
        lc1, lc2 = st.columns([3, 1])
        legacy_model = lc1.text_input("Modello della cache legacy", value=cfg.model)
        if lc2.button("Importa", width="stretch"):
            with common.guarded("Import cache legacy", model=legacy_model):
                with st.spinner("Conversione…"):
                    n = import_legacy_cache(slug, legacy_model)
                st.success(f"Importati {n:,} vettori per {legacy_model}")
