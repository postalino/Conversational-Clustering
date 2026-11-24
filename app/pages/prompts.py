import streamlit as st

import common
from labeling import build_inputs, keywords_by_cluster, name_cluster, preview
from prompts import DEFAULTS, get_prompt, missing_variables, reset_prompt, save_prompt

st.title("📝 Prompt")
st.caption("Tutti i prompt usati dal sistema, in un unico posto. Le modifiche sono salvate in `data/prompts.json`; "
           "ogni set salvato registra la versione dei prompt con cui sono stati generati i nomi.")

ss = st.session_state
pid = st.selectbox("Prompt", list(DEFAULTS), format_func=lambda p: DEFAULTS[p].title)
spec = DEFAULTS[pid]
prompt = get_prompt(pid)

c1, c2 = st.columns([3, 1])
c1.markdown(f"**A cosa serve.** {spec.purpose}")
c2.markdown(f"Versione `{prompt.version}`  \n" + ("✏️ **personalizzato**" if prompt.customized else "default"))
st.caption("Variabili disponibili: " + " ".join(f"`${v}`" for v in spec.variables))

with st.form(f"form_{pid}"):
    system = st.text_area("System", value=prompt.system, height=320)
    user = st.text_area("User (template)", value=prompt.user, height=220, disabled=not spec.user)
    s1, s2 = st.columns(2)
    saved = s1.form_submit_button("💾 Salva", type="primary", width="stretch")
    reset = s2.form_submit_button("↩️ Ripristina default", width="stretch")
if saved:
    missing = missing_variables(pid, system, user if spec.user else "")
    with common.guarded("Salvataggio prompt", prompt=pid):
        save_prompt(pid, system, user if spec.user else "")
        st.success(f"Salvato (versione {get_prompt(pid).version})")
        if missing:
            st.warning("Il testo non usa più: " + ", ".join(f"`${v}`" for v in missing)
                       + " — quelle informazioni non arriveranno al modello.")
if reset:
    reset_prompt(pid)
    st.rerun()

# ----------------------------------------------------------------------------- anteprima e prova
if pid == "cluster_naming":
    st.divider()
    st.subheader("Anteprima e prova su un cluster reale")
    r = ss.get("result")
    if r is None:
        st.caption("Esegui un clustering per vedere il prompt completo con dati reali.")
    else:
        res, clusters, df = r["res"], r["clusters"], ss.sample_df
        cid = st.selectbox("Cluster", res.cluster_ids, format_func=lambda c: f"Cluster {c} ({clusters[str(c)]['size']})")
        lang = st.radio("Lingua", list(common.LANGUAGES), horizontal=True,
                        index=common.language_index(ss.label_language))
        # il materiale per cluster dipende solo dal risultato (fisso): calcolato una volta e riusato
        inputs_cache = r.setdefault("_naming_inputs", {})
        if cid not in inputs_cache:
            inputs_cache[cid] = build_inputs(r["Xp"], res.labels, df["Text"].tolist(), [cid],
                                             keywords_by_cluster(clusters))[0]
        item = inputs_cache[cid]
        with st.expander("Messaggi inviati al modello", expanded=False):
            st.code(preview(item, lang), language="json")
        if st.button(f"Prova con `{common.chat_cfg().model}`"):
            with common.guarded("Prova prompt", prompt=pid, cluster=cid):
                with st.spinner("…"):
                    out = name_cluster(common.llm_service(common.chat_cfg()), item, lang)
                st.json(out)
