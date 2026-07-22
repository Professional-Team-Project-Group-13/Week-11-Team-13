"""Admin portal: analytics, cross-domain, account management, login records."""
import streamlit as st

import config
from backend import auth, database
from frontend import theme


def render():
    s = database.case_stats()
    cols = st.columns(5)
    cols[0].markdown(theme.kpi(s["total"], "Cases"), unsafe_allow_html=True)
    cols[1].markdown(theme.kpi(s["ade"], "ADE flagged"), unsafe_allow_html=True)
    cols[2].markdown(theme.kpi(s["closed"], "Closed"), unsafe_allow_html=True)
    cols[3].markdown(theme.kpi(auth.user_count(), "Users"), unsafe_allow_html=True)
    cols[4].markdown(theme.kpi(f'{s["referral_rate"]:.0f}%', "ADE rate"), unsafe_allow_html=True)
    st.write("")

    tab_users, tab_logins, tab_research, tab_audit = st.tabs(
        ["Accounts", "Login records", "Cross-domain", "Audit log"])

    # ---------------- account management ----------------
    with tab_users:
        pending = auth.list_pending()
        st.markdown(f'<div class="card"><span class="tag">Staff access requests</span> '
                    f'<b style="margin-left:8px">{len(pending)} pending</b>',
                    unsafe_allow_html=True)
        if not pending:
            st.caption("No pending requests.")
        for u in pending:
            c = st.columns([3, 1, 1])
            c[0].markdown(f'**{u["full_name"] or u["username"]}** · `{u["username"]}` '
                          f'· requested **{u["role"]}** · {u["created_at"][:16].replace("T"," ")}')
            if c[1].button("Approve", key=f"appr_{u['username']}", type="primary"):
                auth.set_status(u["username"], "active")
                database.log_action(st.session_state.user["username"], "admin",
                                    "approve_staff", detail=u["username"])
                st.success(f"Approved {u['username']}."); st.rerun()
            if c[2].button("Reject", key=f"rej_{u['username']}"):
                auth.delete_user(u["username"])
                database.log_action(st.session_state.user["username"], "admin",
                                    "reject_staff", detail=u["username"])
                st.warning(f"Rejected {u['username']}."); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><span class="tag">Create account</span>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        nu = c1.text_input("Username", key="ad_user")
        fn = c2.text_input("Full name", key="ad_fn")
        c3, c4, c5 = st.columns(3)
        role = c3.selectbox("Role", config.ROLES, key="ad_role")
        p1 = c4.text_input("Password", type="password", key="ad_p1")
        p2 = c5.text_input("Confirm", type="password", key="ad_p2")
        if st.button("Create account", type="primary"):
            if p1 != p2:
                st.error("Passwords do not match.")
            else:
                ok, msg = auth.create_user(nu, p1, role, fn)
                if ok:
                    database.log_action(st.session_state.user["username"], "admin",
                                        "create_user", detail=f"{nu} ({role})")
                    st.success(msg); st.rerun()
                else:
                    st.error(msg)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><span class="tag">Existing users</span>', unsafe_allow_html=True)
        users = auth.list_users()
        st.dataframe(users, use_container_width=True, hide_index=True)
        with st.expander("Reset password / delete user"):
            names = [u["username"] for u in users]
            target = st.selectbox("User", names, key="mg_target")
            np1 = st.text_input("New password", type="password", key="mg_np")
            mc1, mc2 = st.columns(2)
            if mc1.button("Reset password"):
                ok, msg = auth.change_password(target, np1)
                if ok:
                    database.log_action(st.session_state.user["username"], "admin",
                                        "reset_password", detail=target)
                (st.success if ok else st.error)(msg)
            if mc2.button("Delete user"):
                if target == st.session_state.user["username"]:
                    st.error("You cannot delete your own account while signed in.")
                else:
                    auth.delete_user(target)
                    database.log_action(st.session_state.user["username"], "admin",
                                        "delete_user", detail=target)
                    st.warning(f"Deleted {target}."); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- login records ----------------
    with tab_logins:
        st.markdown('<div class="card"><span class="tag">Login records</span>', unsafe_allow_html=True)
        logins = database.recent_logins(200)
        ok = sum(1 for r in logins if r["success"])
        bad = len(logins) - ok
        lc = st.columns(3)
        lc[0].markdown(theme.kpi(len(logins), "Attempts (recent)"), unsafe_allow_html=True)
        lc[1].markdown(theme.kpi(ok, "Successful"), unsafe_allow_html=True)
        lc[2].markdown(theme.kpi(bad, "Failed"), unsafe_allow_html=True)
        st.write("")
        view = [{"time": r["ts"].replace("T", " "), "username": r["username"],
                 "role": r["role"], "result": "✓ success" if r["success"] else "✗ failed",
                 "detail": r["detail"]} for r in logins]
        if view:
            st.dataframe(view, use_container_width=True, hide_index=True)
        else:
            st.caption("No login attempts recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- cross-domain research ----------------
    with tab_research:
        cd = config.CROSS_DOMAIN
        st.markdown('<div class="card"><span class="tag">Research · cross-domain</span>'
                    '<h3>Detection performance by domain</h3>', unsafe_allow_html=True)
        for m in cd["models"]:
            colour = "#E0930C" if "informal" in m["domain"] else theme.TEAL
            pct = m["f1"] * 100
            st.markdown(
                f'<div style="margin:8px 0"><div style="display:flex;justify-content:space-between;font-size:12px">'
                f'<span><b>{m["name"]}</b> <span class="mono note">{m["domain"]}</span></span>'
                f'<span class="mono">{m["f1"]:.2f}</span></div>'
                f'<div style="height:20px;background:#0E2529;border:1px solid rgba(255,255,255,.1);border-radius:6px">'
                f'<div style="height:100%;width:{pct:.0f}%;background:{colour};border-radius:5px"></div></div></div>',
                unsafe_allow_html=True)
        formal = next(m for m in cd["models"] if "formal" in m["domain"])
        informal = next(m for m in cd["models"] if "informal" in m["domain"])
        gap = (formal["f1"] - informal["f1"]) * 100
        st.info(f"{gap:.0f}-point F1 drop from formal to informal text. "
                f"Driver-word overlap across domains: {cd['overlap']*100:.0f}%.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- audit ----------------
    with tab_audit:
        st.markdown('<div class="card"><span class="tag">Audit log</span>', unsafe_allow_html=True)
        audit = database.recent_audit(120)
        if audit:
            st.dataframe(audit, use_container_width=True, hide_index=True)
        else:
            st.caption("No activity yet.")
        st.markdown('</div>', unsafe_allow_html=True)
