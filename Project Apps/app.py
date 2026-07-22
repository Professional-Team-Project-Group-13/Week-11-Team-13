"""
Clinical Triage Console (Streamlit, multi-portal).

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import streamlit as st

import config
from backend import auth, database
from frontend import admin, doctor, login, nurse, patient, theme

st.set_page_config(page_title="SafetyNet AI", page_icon="🩺", layout="wide")

# DB init is idempotent (CREATE TABLE IF NOT EXISTS) — safe to run every rerun.
database.init_db()
auth.seed_accounts()
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.consented = False
    st.session_state.user = None

theme.inject_css()

PORTALS = {"patient": patient, "nurse": nurse, "doctor": doctor, "admin": admin}
CHECKLIST = [
    "**No identifiable data.** I will not enter names, NHS numbers, dates of birth or contact details.",
    "**Decision support only.** I understand this is a research prototype, not a medical device.",
    "**Lawful processing.** I understand data is handled under UK GDPR / DPA 2018, in-session only.",
    "**Authorised use.** I am authorised to use this tool for adverse-event triage.",
]


def privacy_gate():
    theme.appbar(None)
    col = st.columns([1, 1.3, 1])[1]
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Before you begin — privacy checklist")
        st.caption("UK GDPR · Data Protection Act 2018 — confirm each point to continue.")
        checks = [st.checkbox(item, key=f"ck{i}") for i, item in enumerate(CHECKLIST)]
        done = sum(checks)
        st.progress(done / len(CHECKLIST), text=f"{done} of {len(CHECKLIST)} confirmed")
        if st.button("Enter console", type="primary", disabled=done != len(CHECKLIST),
                     use_container_width=True):
            st.session_state.consented = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def sidebar(user):
    with st.sidebar:
        st.markdown(f"### {config.APP_NAME}")
        st.caption(config.APP_TAGLINE)
        st.markdown(f"**{user['full_name']}**  \n`{user['role'].title()} portal`")
        st.divider()
        st.markdown("Engine: **" + ("BioBERT" if _live() else "Preview (lexicon)") + "**")
        st.divider()
        if st.button("Sign out"):
            database.log_action(user["username"], user["role"], "logout")
            for k in ("user", "consented"):
                st.session_state[k] = None if k == "user" else False
            st.rerun()


def _live():
    try:
        from ai import biobert
        return biobert.is_live()
    except Exception:                              # noqa: BLE001
        return False


def main():
    if not st.session_state.get("consented"):
        privacy_gate()
        return
    user = st.session_state.get("user")
    if not user:
        login.render()
        return
    sidebar(user)
    theme.appbar(user)
    theme.status_banner(_live())
    # Access guard: a user can only ever reach their own role's portal.
    portal = PORTALS.get(user["role"])
    if portal is None:
        st.error("Your account role has no portal assigned. Contact an administrator.")
        return
    portal.render()


main()
