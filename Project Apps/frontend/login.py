"""Login: Patient entrance and Clinician/Staff entrance (sign in + request)."""
import streamlit as st

import config
from backend import auth, database
from frontend import theme

STAFF_ROLES = {"nurse", "doctor", "admin"}


def _do_signin(user, allowed, wrong_entrance_msg):
    if user is None:
        st.error("Invalid username or password.")
        return
    if user.get("status") == "pending":
        st.warning("Your staff account is awaiting administrator approval. "
                   "You'll be able to sign in once it's approved.")
        return
    if user["role"] not in allowed:
        st.error(wrong_entrance_msg)
        return
    st.session_state.user = user
    st.rerun()


def render():
    theme.appbar(None)
    col = st.columns([1, 1.2, 1])[1]
    with col:
        entrance = st.tabs(["🧑 Patient", "🩺 Clinician / Staff"])

        # ---------------- PATIENT ENTRANCE ----------------
        with entrance[0]:
            if "pt_view" not in st.session_state:
                st.session_state.pt_view = "signin"

            if config.ALLOW_PATIENT_SIGNUP:
                seg1, seg2 = st.columns(2)
                if seg1.button("Sign in", use_container_width=True,
                               type="primary" if st.session_state.pt_view == "signin" else "secondary"):
                    st.session_state.pt_view = "signin"
                    st.rerun()
                if seg2.button("Create account", use_container_width=True,
                               type="primary" if st.session_state.pt_view == "signup" else "secondary"):
                    st.session_state.pt_view = "signup"
                    st.rerun()

            if st.session_state.pt_view == "signin":
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Patient sign in")
                st.caption("For people reporting how a medication has affected them.")
                default_user = st.session_state.pop("just_registered_username", "")
                u = st.text_input("Username", value=default_user, key="pt_user")
                p = st.text_input("Password", type="password", key="pt_pass")
                if st.button("Sign in as patient", type="primary", use_container_width=True):
                    _do_signin(auth.authenticate(u, p), {"patient"},
                               "That's a staff account. Please use the "
                               "‘Clinician / Staff’ tab.")
                st.markdown('</div>', unsafe_allow_html=True)

            elif config.ALLOW_PATIENT_SIGNUP and st.session_state.pt_view == "signup":
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Create a patient account")
                st.caption("Patients only. Staff accounts are requested on the other tab.")
                fn = st.text_input("Your name", key="rg_fn")
                nu = st.text_input("Choose a username", key="rg_user")
                p1 = st.text_input("Password", type="password", key="rg_p1")
                p2 = st.text_input("Confirm password", type="password", key="rg_p2")
                if st.button("Create patient account", use_container_width=True):
                    if p1 != p2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = auth.create_user(nu, p1, "patient", fn)
                        if ok:
                            database.log_action(nu, "patient", "register")
                            st.session_state.just_registered_username = nu
                            st.session_state.pt_view = "signin"
                            st.success("Account created! Please sign in.")
                            st.rerun()
                        else:
                            st.error(msg)
                st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- CLINICIAN / STAFF ENTRANCE ----------------
        with entrance[1]:
            s_in, s_req = st.tabs(["Sign in", "Request access"])
            with s_in:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Clinician / staff sign in")
                st.caption("Nurse · Doctor · Admin.")
                u = st.text_input("Username", key="st_user")
                p = st.text_input("Password", type="password", key="st_pass")
                if st.button("Sign in as staff", type="primary", use_container_width=True):
                    _do_signin(auth.authenticate(u, p), STAFF_ROLES,
                               "That's a patient account. Please use the ‘Patient’ tab.")
                st.markdown('</div>', unsafe_allow_html=True)
            with s_req:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Request a staff account")
                st.caption("Choose your role. Your request must be approved by an "
                           "administrator before you can sign in.")
                fn = st.text_input("Full name", key="sr_fn")
                nu = st.text_input("Choose a username", key="sr_user")
                role = st.selectbox("Role", ["nurse", "doctor"], key="sr_role",
                                    help="Nurse: confirm/escalate. Doctor: full clinical review.")
                p1 = st.text_input("Password", type="password", key="sr_p1")
                p2 = st.text_input("Confirm password", type="password", key="sr_p2")
                if st.button("Submit request", use_container_width=True):
                    if p1 != p2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = auth.request_staff_account(nu, p1, role, fn)
                        if ok:
                            database.log_action(nu, role, "request_staff_account",
                                                detail=f"{role} (pending)")
                            st.success("Request submitted. An administrator will review "
                                       "and approve your account.")
                        else:
                            st.error(msg)
                st.markdown('</div>', unsafe_allow_html=True)