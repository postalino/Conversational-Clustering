import pandas as pd
import plotly.express as px
import streamlit as st

import common
from config import get_env
from dataset import DatasetDownloader, SampleConfig

st.title("🗂️ Dataset")

dl = common.downloader()
meta = dl.metadata()

# ----------------------------------------------------------------------------- download
with st.container(border=True):
    st.subheader("1 · Sorgente")
    c1, c2 = st.columns([3, 1])
    slug = c1.text_input("Dataset Kaggle (owner/nome)", value=meta.get("dataset") or get_env("KAGGLE_DATASET"))
    force = c2.checkbox("Forza nuovo download", help="Riscarica anche se il CSV è già presente")
    if meta:
        st.success(f"Dataset presente: **{meta.get('dataset')}** · file `{meta.get('file')}`"
                   + (f" · scaricato {meta['downloaded']}" if meta.get("downloaded") else ""), icon="✅")
    else:
        st.info("Nessun dataset scaricato.", icon="ℹ️")
    if st.button("Scarica da Kaggle", type="primary"):
        with common.guarded("Download dataset", dataset=slug):
            with st.spinner(f"Download di {slug}…"):
                DatasetDownloader(dataset_name=slug).download_data(force=force)
            # nuovo dataset: niente di quanto è in sessione è più valido (testi, vettori, risultati)
            common.set_workspace(st.session_state.sample_cfg, None, clear_caches=True)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Download completato")
            st.rerun()

if not dl.is_downloaded():
    st.stop()

raw = None
with common.guarded("Lettura dataset"):
    raw = common.load_raw()
if raw is None:
    st.stop()

with st.expander(f"Anteprima dataset completo — {len(raw):,} righe × {raw.shape[1]} colonne"):
    st.dataframe(raw.head(200), width="stretch", height=300)

# ----------------------------------------------------------------------------- campione
with st.container(border=True):
    st.subheader("2 · Campione di lavoro")
    st.caption("Il campione determina quali testi verranno incorporati e clusterizzati. "
               "Stesso seed + stessi filtri = stesso campione (riproducibile).")
    cfg: SampleConfig = st.session_state.sample_cfg
    cols = list(raw.columns)
    text_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(raw[c])] or cols
    # Solo colonne intere: la cache degli embedding indicizza per id numerico (l'univocità è verificata dopo)
    id_opts = ["(indice di riga)"] + [c for c in cols if pd.api.types.is_integer_dtype(raw[c])]

    c1, c2, c3 = st.columns(3)
    text_col = c1.selectbox("Colonna testo", text_cols,
                            index=text_cols.index(cfg.text_col) if cfg.text_col in text_cols else 0)
    id_sel = c2.selectbox("Colonna id", id_opts, index=id_opts.index(cfg.id_col) if cfg.id_col in id_opts else 0)
    extra = c3.multiselect("Colonne extra da mostrare", [c for c in cols if c not in (text_col, id_sel)],
                           default=[c for c in cfg.extra_cols if c in cols and c not in (text_col, id_sel)])

    c1, c2, c3, c4, c5 = st.columns(5)
    use_all = c1.checkbox("Usa tutto", value=cfg.sample_size is None)
    size = c1.number_input("Dimensione campione", 50, len(raw), value=min(cfg.sample_size or 2000, len(raw)),
                           step=500, disabled=use_all)
    seed = c2.number_input("Seed", 0, 10_000, value=cfg.seed)
    min_chars = c3.number_input("Min caratteri", 0, 10_000, value=cfg.min_chars)
    max_on = c4.checkbox("Limita lunghezza max", value=cfg.max_chars is not None)
    max_chars = c4.number_input("Max caratteri", 50, 100_000, value=cfg.max_chars or 2000, disabled=not max_on)
    dedupe = c5.checkbox("Rimuovi testi duplicati", value=cfg.dedupe,
                         help="Il dataset Amazon contiene molte recensioni identiche ripetute su prodotti diversi")
    n_cached = common.current_store(text_col).count()
    only_cached = c5.checkbox(f"Solo testi già in cache ({n_cached:,})", value=cfg.only_cached and n_cached > 0,
                              disabled=n_cached == 0,
                              help=f"Campiona solo tra i testi con embedding già calcolati per "
                                   f"{common.emb_cfg().model}: nessun costo API")

    new_cfg = SampleConfig(
        text_col=text_col, id_col=None if id_sel == "(indice di riga)" else id_sel,
        sample_size=None if use_all else int(size), seed=int(seed), min_chars=int(min_chars),
        max_chars=int(max_chars) if max_on else None, dedupe=dedupe, extra_cols=extra,
        only_cached=only_cached,
    )
    if st.button("Crea campione", type="primary"):
        with common.guarded("Creazione campione", config=new_cfg.to_dict()):
            df = common.build_sample(new_cfg)
            if len(df) < 10:
                raise ValueError(f"Il campione ha solo {len(df)} testi dopo i filtri: allenta i filtri "
                                 "(lunghezza minima/massima, solo testi in cache).")
            common.set_workspace(new_cfg, df)
            if common.load_embeddings_for_sample():
                st.success(f"Campione di {len(df):,} testi creato — embedding già tutti in cache ✅")
            else:
                st.success(f"Campione di {len(df):,} testi creato. Prossimo passo: **Embedding**.")

# ----------------------------------------------------------------------------- statistiche
df = st.session_state.sample_df
if df is not None:
    with st.container(border=True):
        st.subheader("3 · Campione corrente")
        lengths = df["Text"].str.len()
        c = st.columns(4)
        c[0].metric("Testi", f"{len(df):,}")
        c[1].metric("Lunghezza media", f"{lengths.mean():.0f} car.")
        c[2].metric("Mediana", f"{lengths.median():.0f} car.")
        c[3].metric("Token stimati", f"{int(lengths.sum() / 4):,}")
        t1, t2 = st.tabs(["Distribuzione lunghezze", "Testi"])
        with t1:
            # la figura dipende solo dal campione: si ricostruisce solo quando il campione cambia
            cached = st.session_state.get("_len_hist")
            if cached is None or cached[0] is not df:
                fig = px.histogram(lengths.clip(upper=lengths.quantile(0.99)), nbins=60,
                                   labels={"value": "caratteri"}).update_layout(showlegend=False, height=300)
                st.session_state["_len_hist"] = cached = (df, fig)
            st.plotly_chart(cached[1], width="stretch")
            if "Score" in df.columns:
                st.bar_chart(df["Score"].value_counts().sort_index(), height=200)
        with t2:
            st.dataframe(df, width="stretch", height=400)
