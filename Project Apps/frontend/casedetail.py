"""Shared case-detail / review panel.

Option B access split:
  * NURSE  -> lightweight confirm / escalate (prediction + triage only)
  * DOCTOR -> full clinical review with explainability (LIME/SHAP) + precedents
"""
import json

import streamlit as st

from backend import database, reports
from frontend import theme

FULL_REVIEW_ROLES = {"doctor", "admin"}   # who sees explainability + precedents


def render_case_detail(case, role):
    a = {}
    if case.get("analysis_json"):
        try:
            a = json.loads(case["analysis_json"])
        except Exception:                          # noqa: BLE001
            a = {}
    colour = theme.priority_colour(case.get("priority", "P3"))
    text = ((case.get("symptoms") or "") + " " + (case.get("medication") or "")).strip()

    # ---- summary (everyone who can open a case sees this) ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<span class="tag">Case #{case["id"]}</span>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown(theme.gauge_html((case.get("proba") or 0) * 100,
                                     case.get("triage_band", "—"), colour),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'**Priority:** {theme.band_pill(case.get("priority","—"), colour)} · '
            f'**P(ADE):** <span class="mono">{case.get("proba",0):.2f}</span><br>'
            f'**Prediction:** {"ADE" if case.get("label") else "No ADE"} · '
            f'**Severity:** {str(case.get("severity","-")).title()}<br>'
            f'**Recommended:** {case.get("action","-")}<br>'
            f'**Domain:** {case.get("domain","-")}',
            unsafe_allow_html=True)
        if case.get("red_flag"):
            st.error("Red-flag symptom present — urgent.")
    st.markdown('**Symptoms:** ' + (case.get("symptoms") or "—"))
    st.markdown('**Medication:** ' + (case.get("medication") or "—"))
    st.markdown(f'**Vitals:** HR {case.get("heart_rate") or "—"}, '
                f'BP {case.get("systolic_bp") or "—"}, Temp {case.get("temperature") or "—"}, '
                f'SpO₂ {case.get("spo2") or "—"} · **Age:** {case.get("age") or "—"}')
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- explainability + precedents: DOCTOR / clinician only ----
    if role in FULL_REVIEW_ROLES:
        if a.get("lime"):
            st.markdown('<div class="card"><span class="tag">Explainability · clinician review</span>',
                        unsafe_allow_html=True)
            if text:
                st.markdown(theme.sentence_html(text, a["lime"]), unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            cc1.markdown(theme.driver_bars_html(a["lime"], "LIME"), unsafe_allow_html=True)
            cc2.markdown(theme.driver_bars_html(a.get("shap", a["lime"]), "SHAP"),
                         unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        if a.get("neighbours"):
            st.markdown('<div class="card"><span class="tag">FAISS precedents</span>',
                        unsafe_allow_html=True)
            st.markdown(theme.precedents_html(a["neighbours"]), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    elif role == "nurse":
        st.markdown('<div class="card note">Confirm whether this is a genuine adverse '
                    'event, or escalate to a clinician for the detailed explainability '
                    'review.</div>', unsafe_allow_html=True)


def decision_controls(case, role, user):
    st.markdown('<div class="card"><span class="tag">Decision</span>', unsafe_allow_html=True)
    note = st.text_area("Clinical note", value=case.get("clinician_note") or "",
                        key=f"note_{case['id']}")
    now = __import__("datetime").datetime.utcnow().isoformat(timespec="seconds")
    cols = st.columns(4)

    # NURSE: confirm (close) or escalate to a clinician
    if role == "nurse":
        if cols[0].button("Confirm & close", key=f"close_{case['id']}", type="primary"):
            database.update_case(case["id"], status="closed", clinician_note=note,
                                 clinician_username=user["username"], decided_at=now)
            database.log_action(user["username"], role, "nurse_confirm_close", case["id"])
            st.success("Confirmed and closed."); st.rerun()
        if cols[1].button("Escalate to clinician", key=f"esc_{case['id']}"):
            database.update_case(case["id"], status="escalated", clinician_note=note,
                                 clinician_username=user["username"], decided_at=now)
            database.log_action(user["username"], role, "escalate_case", case["id"])
            st.warning("Escalated to clinician."); st.rerun()
        if cols[2].button("Save note", key=f"save_{case['id']}"):
            database.update_case(case["id"], clinician_note=note); st.toast("Saved.")

    # DOCTOR / clinician: final decision + PDF report
    else:
        if cols[0].button("Confirm & close", key=f"close_{case['id']}", type="primary"):
            database.update_case(case["id"], status="closed", clinician_note=note,
                                 clinician_username=user["username"], decided_at=now)
            database.log_action(user["username"], role, "clinician_decision", case["id"])
            st.success("Reviewed and closed."); st.rerun()
        if cols[1].button("Save note", key=f"save_{case['id']}"):
            database.update_case(case["id"], clinician_note=note); st.toast("Saved.")
        if cols[2].button("PDF report", key=f"pdf_{case['id']}"):
            c2 = database.get_case(case["id"]); c2["clinician_note"] = note
            path = reports.generate_case_report(c2)
            with open(path, "rb") as f:
                st.download_button("Download report", f.read(),
                                   file_name=f"case_{case['id']}.pdf",
                                   mime="application/pdf", key=f"dl_{case['id']}")
    st.markdown('</div>', unsafe_allow_html=True)
