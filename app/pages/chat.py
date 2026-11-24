import streamlit as st

import common
from chatbot import build_messages
from runs import get_selected, load_run

st.title("💬 Chatbot")
st.caption("Versione provvisoria: solo conversazione, nessuna azione sui cluster (la parte agentica arriverà dopo).")

ss = st.session_state
selected = get_selected()

with st.sidebar:
    st.caption("CHATBOT")
    use_ctx = st.toggle("Includi il set iniziale come contesto", value=selected is not None, disabled=selected is None,
                        help="Passa all'LLM nomi, descrizioni, dimensioni e parole chiave dei cluster scelti")
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.3, 0.1)
    if st.button("🧹 Nuova conversazione", width="stretch"):
        ss.chat_history = []
        st.rerun()

cfg = common.chat_cfg()
st.caption(f"Modello: `{cfg.model}` ({cfg.provider})"
           + (f" · contesto: `{selected}`" if use_ctx and selected else ""))

run = None
if use_ctx and selected:
    with common.guarded("Caricamento contesto set iniziale"):
        run = load_run(selected, with_arrays=False)

for msg in ss.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Scrivi un messaggio…"):
    ss.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        reply = None
        with common.guarded("Risposta chatbot", model=cfg.model):
            stream = common.llm_service(cfg).stream_chat(build_messages(ss.chat_history, run), temperature=temperature)
            reply = st.write_stream(stream)
        if reply:
            ss.chat_history.append({"role": "assistant", "content": reply})
        else:
            ss.chat_history.pop()  # il messaggio fallito non entra nella cronologia
