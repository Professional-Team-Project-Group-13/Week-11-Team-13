"""Doctor portal: escalated + high-priority cases, final decision, reports."""
import streamlit as st

from backend import database
from frontend import casedetail, theme


def render():
    user = st.session_state.user
    s = database.case_stats()
    cols = st.columns(4)
    cols[0].markdown(theme.kpi(s["escalated"], "Escalated to me"), unsafe_allow_html=True)
    cols[1].markdown(theme.kpi(s["p1"], "P1 urgent"), unsafe_allow_html=True)
    cols[2].markdown(theme.kpi(s["closed"], "Closed"), unsafe_allow_html=True)
    cols[3].markdown(theme.kpi(s["total"], "Total"), unsafe_allow_html=True)
    st.write("")

    st.caption("Clinical review with full explainability (LIME/SHAP) and precedents.")
    escalated = database.list_cases(status="escalated")
    urgent = [c for c in database.list_cases(status="pending") if c["priority"] == "P1"]
    queue = escalated + urgent
    st.markdown(f'<div class="card"><span class="tag">Doctor review</span> '
                f'<b style="margin-left:8px">{len(queue)} cases</b>', unsafe_allow_html=True)
    if not queue:
        st.info("No escalated or urgent cases.")
    for c in queue:
        with st.expander(f'#{c["id"]}  [{c["priority"]}]  {c["status"]}  ·  {c["symptoms"][:60]}'):
            casedetail.render_case_detail(c, "doctor")
            casedetail.decision_controls(c, "doctor", user)
    st.markdown('</div>', unsafe_allow_html=True)
