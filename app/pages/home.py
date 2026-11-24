import streamlit as st

import common
from config import get_env, mask_secret
from logging_setup import ERRORS
from runs import get_selected, list_runs, load_run

st.title("🧭 Conversational Clustering")
st.caption("Preparazione del set iniziale: dataset → embedding → clustering → scelta del set di partenza per l'agente.")

ss = st.session_state
dl = common.downloader()
store = common.current_store()
runs = list_runs()
selected = get_selected()

# ----------------------------------------------------------------------------- pipeline
ready = common.workspace_ready()
steps = [
    ("Dataset", dl.is_downloaded(), dl.slug if dl.is_downloaded() else "da scaricare", "pages/data.py"),
    ("Campione", ss.sample_df is not None,
     f"{len(ss.sample_df):,} testi" if ss.sample_df is not None else "da creare", "pages/data.py"),
    ("Embedding", ready,
     f"{ss.X.shape[0]:,} × {ss.X.shape[1]} · {ss.X_model}" if ready
     else f"{store.count():,} in cache per {store.model}", "pages/embed.py"),
    ("Clustering", len(runs) > 0 or ss.result is not None, f"{len(runs)} set salvati", "pages/cluster.py"),
    ("Set iniziale", selected is not None, selected or "non scelto", "pages/runs.py"),
]
cols = st.columns(len(steps))
for col, (title, done, detail, page) in zip(cols, steps):
    with col.container(border=True):
        st.markdown(f"{'✅' if done else '⬜'} **{title}**")
        st.caption(detail)
        st.page_link(page, label="Apri", icon="➡️")

if selected:
    run = load_run(selected, with_arrays=False)
    with st.container(border=True):
        st.markdown(f"#### 📌 Set iniziale scelto: {run.name}")
        st.caption(f"{run.algorithm} · {run.metrics.get('n_clusters')} cluster · silhouette "
                   f"{common.fmt(run.metrics.get('silhouette'))} · `{run.embedding_model}`")
        names = [f"**{v.get('name') or 'Cluster ' + c}** ({v.get('size')})" for c, v in run.clusters.items()]
        st.markdown(" · ".join(names[:30]))

# ----------------------------------------------------------------------------- diagnostica
st.subheader("Diagnostica")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.markdown("**Kaggle**")
    user, key = get_env("KAGGLE_USERNAME"), get_env("KAGGLE_KEY")
    st.markdown(f"{'✅' if user and key else '❌'} credenziali {'presenti' if user and key else 'mancanti nel .env'}")
    st.caption(f"utente: {user or '—'} · key: {mask_secret(key)}")
for col, kind, cfg in ((c2, "chat", ss.chat_cfg), (c3, "embedding", ss.emb_cfg)):
    with col.container(border=True):
        st.markdown(f"**{'LLM chat' if kind == 'chat' else 'Embedding'}** · {cfg.provider}")
        st.caption(f"`{cfg.model}` · {cfg.base_url} · key {mask_secret(cfg.api_key)}")
        st.page_link("pages/settings.py", label="Configura e testa", icon="⚙️")

errors = list(ERRORS)  # copia istantanea della coda condivisa
if errors:
    last = errors[-1]
    st.warning(f"Ultimo errore ({last.time:%H:%M:%S}) in **{last.step}**: {last.error_type}: {last.message[:200]}",
               icon="🚨")
    st.page_link("pages/logs.py", label="Vedi tutti gli errori", icon="🩺")
