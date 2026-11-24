"""Entry point della GUI.  Avvio:  streamlit run app/app.py"""

import streamlit as st

st.set_page_config(page_title="Conversational Clustering", page_icon="🧭", layout="wide")

import common  # noqa: E402  (configura sys.path e logging)
from projection import warmup_umap  # noqa: E402
from runs import get_selected  # noqa: E402

common.init_state()
warmup_umap()  # compila UMAP in background (una volta per processo) mentre si lavora sulle altre pagine

pages = {
    "Pipeline": [
        st.Page("pages/home.py", title="Panoramica", icon="🧭", default=True),
        st.Page("pages/data.py", title="Dataset", icon="🗂️"),
        st.Page("pages/embed.py", title="Embedding", icon="🧬"),
        st.Page("pages/cluster.py", title="Clustering", icon="🔬"),
        st.Page("pages/runs.py", title="Set iniziali", icon="📌"),
    ],
    "Assistente": [st.Page("pages/chat.py", title="Chatbot", icon="💬")],
    "Sistema": [
        st.Page("pages/settings.py", title="Modelli & API", icon="⚙️"),
        st.Page("pages/prompts.py", title="Prompt", icon="📝"),
        st.Page("pages/logs.py", title="Log & Errori", icon="🩺"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    ss = st.session_state
    st.caption("STATO SESSIONE")
    sample = f"{len(ss.sample_df):,} testi" if ss.sample_df is not None else "—"
    emb = "pronti" if common.workspace_ready() else "—"
    st.markdown(
        f"**LLM** `{ss.chat_cfg.model}`  \n"
        f"**Embedding** `{ss.emb_cfg.model}`  \n"
        f"**Campione** {sample} · embedding {emb}"
    )
    sel = get_selected()
    st.markdown(f"**Set iniziale** {'`' + sel + '`' if sel else '— non scelto'}")
    n_err = common.new_errors_count()
    if n_err > 0:
        st.error(f"{n_err} nuovi errori", icon="🚨")
        st.page_link("pages/logs.py", label="Apri Log & Errori", icon="🩺")

nav.run()
