"""Nurse dashboard: triage queue + review."""
import streamlit as st

from backend import database
from frontend import casedetail, theme


def render():
    user = st.session_state.user
    s = database.case_stats()
    cols = st.columns(5)
    cols[0].markdown(theme.kpi(s["total"], "Total cases"), unsafe_allow_html=True)
    cols[1].markdown(theme.kpi(s["pending"], "Pending"), unsafe_allow_html=True)
    cols[2].markdown(theme.kpi(s["p1"], "P1 urgent"), unsafe_allow_html=True)
    cols[3].markdown(theme.kpi(s["ade"], "ADE flagged"), unsafe_allow_html=True)
    cols[4].markdown(theme.kpi(f'{s["referral_rate"]:.0f}%', "Referral rate"), unsafe_allow_html=True)
    st.write("")

    st.caption("Confirm genuine adverse events, or escalate to a clinician for review.")
    pending = database.list_cases(status="pending")
    st.markdown(f'<div class="card"><span class="tag">Triage queue</span> '
                f'<b style="margin-left:8px">{len(pending)} awaiting review</b>',
                unsafe_allow_html=True)
    if not pending:
        st.info("No pending cases.")
    for c in pending:
        colour = theme.priority_colour(c["priority"])
        label = (f'#{c["id"]} · {theme.band_pill(c["priority"], colour)} · '
                 f'P(ADE) {c["proba"]:.2f} · {c["symptoms"][:70]}…')
        with st.expander(f'#{c["id"]}  [{c["priority"]}]  P(ADE) {c["proba"]:.2f}  ·  {c["symptoms"][:60]}'):
            casedetail.render_case_detail(c, "nurse")
            casedetail.decision_controls(c, "nurse", user)
    st.markdown('</div>', unsafe_allow_html=True)
