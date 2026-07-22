"""Patient self-intake portal: submit, safety banner, history + clinician feedback."""
import datetime
import json

import streamlit as st

from backend import database, prediction, triage
from frontend import theme

# friendly status text shown to patients
_STATUS_TEXT = {
    "pending":   ("Awaiting review", "A clinician will review your report shortly."),
    "escalated": ("With a clinician", "A clinician is carrying out a detailed review."),
    "closed":    ("Reviewed", "A clinician has reviewed your report."),
}


def _urgent_banner(t):
    """Prominent emergency guidance for high-risk / red-flag cases."""
    if t.get("red_flags") or t["priority"] == "P1":
        st.error(
            "⚠️ **Your report suggests a potentially serious reaction.** "
            "If you have trouble breathing, swelling of the face or throat, chest pain, "
            "fainting, or are getting rapidly worse, **do not wait for this review — "
            "seek urgent medical help now (call your local emergency number / 999).**")


def _render_result(a, t, text):
    _urgent_banner(t)
    st.markdown('<div class="card"><span class="tag">Triage outcome</span>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(theme.gauge_html(t["triage_pct"], t["triage_band"], t["band_colour"]),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'**Result:** {"Possible adverse event" if a["proba"] >= 0.5 else "No adverse event detected"}<br>'
            f'**Priority:** {theme.band_pill(t["priority"], t["band_colour"])}<br>'
            f'**What happens next:** your report has been sent to the clinical team for review.',
            unsafe_allow_html=True)
    st.caption("This is an automated triage estimate and is not a diagnosis. "
               "A qualified clinician makes the final decision.")
    st.markdown('</div>', unsafe_allow_html=True)


def _history(user):
    mine = database.list_cases(patient_username=user["username"])
    st.markdown('<div class="card"><span class="tag">My reports</span>', unsafe_allow_html=True)
    if not mine:
        st.caption("You haven't submitted any reports yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    for c in mine[:15]:
        colour = theme.priority_colour(c["priority"])
        label, blurb = _STATUS_TEXT.get(c["status"], (c["status"].title(), ""))
        when = c["created_at"][:16].replace("T", " ")
        st.markdown(
            f'**#{c["id"]}** · {when} · {theme.band_pill(c["priority"], colour)} '
            f'· **{label}**', unsafe_allow_html=True)
        st.markdown(f'<span class="note">{c["symptoms"][:110]}</span>', unsafe_allow_html=True)
        # clinician feedback, once reviewed
        if c["status"] in ("closed", "escalated"):
            note = (c.get("clinician_note") or "").strip()
            msg = blurb + (f' Clinician note: “{note}”' if note else "")
            st.info(msg)
        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,.08)">',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render():
    user = st.session_state.user
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Report how you're feeling")
    st.caption("Describe your symptoms and medication. A clinician will review the result. "
               "Do not enter your name, NHS number, or other identifying details.")
    symptoms = st.text_area("Symptoms / what happened",
                            placeholder="e.g. Since starting the new tablets I've had an itchy rash and felt dizzy and sick.")
    medication = st.text_input("Medication (name & dose if known)", placeholder="e.g. Amoxicillin 500mg")
    c1, c2, c3, c4, c5 = st.columns(5)
    age = c1.number_input("Age", 0, 120, 0)
    hr = c2.number_input("Heart rate", 0, 250, 0)
    bp = c3.number_input("Systolic BP", 0, 300, 0)
    temp = c4.number_input("Temp °C", 0.0, 45.0, 0.0, step=0.1)
    spo2 = c5.number_input("SpO₂ %", 0, 100, 0)
    submit = st.button("Submit for triage", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        text = (symptoms + " " + medication).strip()
        if not text:
            st.error("Please describe your symptoms first.")
        else:
            a = prediction.analyze(text)
            vitals = {"heart_rate": hr or None, "systolic_bp": bp or None,
                      "temperature": temp or None, "spo2": spo2 or None}
            t = triage.triage(a, symptoms=symptoms, age=age or None, vitals=vitals)
            cid = database.create_case({
                "patient_username": user["username"],
                "created_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                "symptoms": symptoms, "medication": medication, "age": age or None,
                "heart_rate": hr or None, "systolic_bp": bp or None,
                "temperature": temp or None, "spo2": spo2 or None,
                "proba": a["proba"], "label": int(a["proba"] >= 0.5), "severity": a["severity"],
                "triage_band": t["triage_band"], "priority": t["priority"], "action": t["action"],
                "domain": a["domain"], "engine": a["engine"],
                "red_flag": int(bool(t["red_flags"])),
                "status": "pending", "analysis_json": json.dumps(a)})
            database.log_action(user["username"], "patient", "submit_case", cid)
            st.success(f"Submitted. Reference #{cid}.")
            _render_result(a, t, text)

    _history(user)
