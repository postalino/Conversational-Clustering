import logging

import pandas as pd
import streamlit as st

import common
from config import LOGS_DIR
from llm import CALL_LOG
from logging_setup import ERRORS, LOG_BUFFER, clear_memory, errors_total, set_level

# Copie istantanee: altri thread/sessioni scrivono di continuo in queste code condivise e iterarle
# direttamente può fallire ("deque mutated during iteration"). list(deque) è atomico.
errors, calls_log, log_entries = list(ERRORS), list(CALL_LOG), list(LOG_BUFFER)

st.title("🩺 Log & Errori")
ss = st.session_state
ss.errors_seen = errors_total()  # aprire questa pagina azzera il badge nella sidebar

tab_err, tab_calls, tab_log, tab_file = st.tabs(
    [f"🚨 Errori ({len(errors)})", f"📡 Chiamate API ({len(calls_log)})", "📜 Log applicazione", "🗄️ File di log"])

# ----------------------------------------------------------------------------- errori
with tab_err:
    if not errors:
        st.success("Nessun errore registrato in questa esecuzione.", icon="✅")
    for e in reversed(errors):
        with st.expander(f"`{e.id}` · {e.time:%H:%M:%S} · **{e.step}** — {e.error_type}: {e.message[:120]}"):
            hint = common.error_hint(Exception(f"{e.error_type} {e.message}"))
            if hint:
                st.info(hint, icon="💡")
            if e.context:
                st.markdown("**Contesto**")
                st.json(e.context)
            st.markdown("**Traceback**")
            st.code(e.traceback, language="python")

# ----------------------------------------------------------------------------- chiamate API
with tab_calls:
    if calls_log:
        calls = pd.DataFrame([c.__dict__ for c in calls_log]).iloc[::-1]
        c = st.columns(4)
        c[0].metric("Chiamate", len(calls))
        c[1].metric("Fallite", int((~calls["ok"]).sum()))
        c[2].metric("Latenza media", f"{calls['latency_ms'].mean():.0f} ms")
        c[3].metric("Token in (totali)", f"{int(calls['prompt_tokens'].fillna(0).sum()):,}")
        only_fail = st.checkbox("Solo fallite")
        if only_fail:
            calls = calls[~calls["ok"]]
        st.dataframe(calls, hide_index=True, width="stretch", column_config={
            "time": st.column_config.DatetimeColumn("ora", format="HH:mm:ss"),
            "latency_ms": st.column_config.NumberColumn("latenza ms", format="%.0f"),
            "ok": st.column_config.CheckboxColumn("ok"),
        })
    else:
        st.caption("Nessuna chiamata API in questa esecuzione.")

# ----------------------------------------------------------------------------- log
with tab_log:
    c1, c2, c3 = st.columns([1, 2, 1])
    level = c1.selectbox("Livello minimo", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    query = c2.text_input("Cerca", placeholder="testo, modulo…")
    if c3.button("Svuota memoria", help="Svuota log ed errori in memoria (il file su disco resta)"):
        clear_memory()
        ss.errors_seen = errors_total()
        st.rerun()
    set_level(logging.DEBUG if level == "DEBUG" else logging.INFO)
    order = logging.getLevelNamesMapping()  # "INFO" -> 20, ...
    entries = [e for e in log_entries if order.get(e.level, 0) >= order[level]
               and (not query or query.lower() in (e.message + e.logger).lower())]
    if entries:
        logs = pd.DataFrame([{"ora": e.time, "livello": e.level, "modulo": e.logger.removeprefix("ccl."),
                              "messaggio": e.message} for e in reversed(entries)])
        st.dataframe(logs.style.map(lambda v: {"ERROR": "color:#d33", "WARNING": "color:#c80",
                                                "CRITICAL": "color:#d33"}.get(v, ""), subset=["livello"]),
                     hide_index=True, width="stretch", height=500,
                     column_config={"ora": st.column_config.DatetimeColumn(format="HH:mm:ss")})
    else:
        st.caption("Nessun messaggio.")

# ----------------------------------------------------------------------------- file
with tab_file:
    path = LOGS_DIR / "app.log"
    if path.exists():
        size = path.stat().st_size
        st.caption(f"`{path}` · {size / 1e3:.0f} kB (include le sessioni precedenti)")
        # file completo letto solo al clic; per la vista basta la coda del file (ultime 300 righe)
        st.download_button("⬇️ Scarica app.log", path.read_bytes, file_name="app.log")
        with path.open("rb") as f:
            f.seek(max(0, size - 256_000))
            tail = f.read().decode("utf-8", errors="replace")
        st.code("\n".join(tail.splitlines()[-300:]) or "(vuoto)", language="log")
    else:
        st.caption("File di log non ancora creato.")
