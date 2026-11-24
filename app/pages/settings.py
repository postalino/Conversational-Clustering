import streamlit as st

import common
from config import mask_secret, save_env_values
from providers import DEFAULT_MODELS, PROVIDERS, ProviderConfig, env_values, list_models, test_connection

st.title("⚙️ Modelli & API")
st.caption("Scegli provider e modello per la chat e per gli embedding. Le modifiche valgono per la sessione; "
           "puoi salvarle come default nel file `.env`.")


@st.cache_data(ttl=600, show_spinner="Recupero elenco modelli…")
def _models(provider: str, base_url: str, api_key: str, kind: str):
    return list_models(ProviderConfig(provider, base_url, api_key, ""), kind)


def provider_panel(kind: str, state_key: str) -> None:
    cfg: ProviderConfig = st.session_state[state_key]
    k = f"{kind}_"
    names = list(PROVIDERS)
    provider = st.selectbox("Provider", names, index=names.index(cfg.provider),
                            format_func=lambda p: PROVIDERS[p]["label"], key=k + "provider")
    same_provider = provider == cfg.provider
    default_url = cfg.base_url if same_provider else PROVIDERS[provider]["base_url"]
    base_url = st.text_input("Base URL", value=default_url, key=f"{k}url_{provider}")
    # La chiave attuale si riusa solo con lo stesso provider: mai inviarla (o salvarla) per un altro servizio.
    current_key = cfg.api_key if same_provider else ""
    if PROVIDERS[provider]["needs_key"]:
        new_key = st.text_input(f"API key (attuale: {mask_secret(current_key)})", type="password",
                                placeholder="lascia vuoto per mantenere quella attuale" if current_key
                                else "inserisci la chiave di questo provider", key=f"{k}key_{provider}")
        api_key = new_key or current_key
    else:
        api_key = ""

    models, err = [], None
    try:
        models = _models(provider, base_url, api_key, kind)
    except Exception as e:  # noqa: BLE001
        err = e
    current = cfg.model if same_provider else DEFAULT_MODELS[(provider, kind)]

    manual = st.toggle("Inserisci il nome del modello a mano", value=not models, key=k + "manual")
    if err is not None:
        st.warning(f"Impossibile ottenere l'elenco modelli: {err}", icon="⚠️")
    if manual or not models:
        model = st.text_input("Modello", value=current, key=f"{k}model_txt_{provider}")
    else:
        query = st.text_input("Filtra modelli", key=f"{k}filter_{provider}", placeholder="es. deepseek, qwen, embed")
        shown = [m for m in models if query.lower() in m.id.lower() or query.lower() in m.name.lower()] or models
        ids = [m.id for m in shown]
        if current not in ids:
            ids.insert(0, current)
        labels = {m.id: m.label for m in shown}
        model = st.selectbox(f"Modello ({len(models)} disponibili)", ids, index=ids.index(current),
                             format_func=lambda i: labels.get(i, i), key=f"{k}model_sel_{provider}")

    new_cfg = ProviderConfig(provider=provider, base_url=base_url, api_key=api_key, model=model)
    c1, c2, c3 = st.columns(3)
    if c1.button("Applica alla sessione", key=k + "apply", type="primary", width="stretch"):
        st.session_state[state_key] = new_cfg
        common.log.info("Config %s aggiornata: %s", kind, new_cfg.to_dict())
        st.toast(f"{kind}: {model}", icon="✅")
        st.rerun()
    if c2.button("Test connessione", key=k + "test", width="stretch"):
        with st.spinner("Chiamata di prova…"):
            ok, msg = test_connection(new_cfg, kind)
        (st.success if ok else st.error)(msg)
    if c3.button("Salva come default (.env)", key=k + "save", width="stretch"):
        with common.guarded("Salvataggio .env"):
            save_env_values(env_values(new_cfg, kind))
            st.session_state[state_key] = new_cfg
            st.success("Salvato in .env")

    if kind == "embedding" and st.session_state.X_model and st.session_state.X_model != model:
        st.info("Cambiando modello di embedding servirà ricalcolare (o caricare dalla cache) gli embedding "
                "del campione: i vettori di modelli diversi non sono confrontabili.", icon="ℹ️")


tab_chat, tab_emb, tab_perf = st.tabs(["💬 Modello chat (LLM)", "🧬 Modello embedding", "⚡ Prestazioni"])
with tab_chat:
    provider_panel("chat", "chat_cfg")
with tab_emb:
    provider_panel("embedding", "emb_cfg")
with tab_perf:
    st.caption("Numero di richieste API contemporanee. Aumentarlo accelera nomi ed embedding finché il provider "
               "non risponde con errori di limite (429): in quel caso il client ritenta da solo, ma conviene ridurlo. "
               "Per Ollama in locale 1-2 è di solito il massimo utile.")
    ss = st.session_state
    p1, p2 = st.columns(2)
    llm_w = p1.slider("Richieste LLM parallele (nomi dei cluster)", 1, 16, ss.llm_workers)
    emb_w = p2.slider("Richieste embedding parallele", 1, 16, ss.emb_workers)
    c1, c2 = st.columns(2)
    if c1.button("Applica alla sessione", key="perf_apply", type="primary", width="stretch"):
        ss.llm_workers, ss.emb_workers = llm_w, emb_w
        st.toast("Impostazioni applicate", icon="✅")
    if c2.button("Salva come default (.env)", key="perf_save", width="stretch"):
        with common.guarded("Salvataggio .env"):
            ss.llm_workers, ss.emb_workers = llm_w, emb_w
            save_env_values({"LLM_CONCURRENCY": str(llm_w), "EMBEDDING_CONCURRENCY": str(emb_w)})
            st.success("Salvato in .env")
