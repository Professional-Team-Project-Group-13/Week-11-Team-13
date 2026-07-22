"""Authentication: PBKDF2 credentials, roles, and pending-approval workflow."""
import hashlib
import os
from datetime import datetime

import config
from backend import database

_ITER = 200_000


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITER).hex()


def _get_user_row(username):
    conn = database.get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return row


def create_user(username, password, role, full_name="", status="active"):
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password are required."
    if role not in config.ROLES:
        return False, "Invalid role."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = os.urandom(16)
    conn = database.get_conn()
    try:
        conn.execute(
            "INSERT INTO users(username,password_hash,salt,role,full_name,status,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (username, _hash(password, salt), salt.hex(), role, full_name, status,
             datetime.utcnow().isoformat(timespec="seconds")))
        conn.commit()
        return True, "Account created."
    except Exception:                              # noqa: BLE001
        return False, "That username already exists."
    finally:
        conn.close()


def request_staff_account(username, password, role, full_name=""):
    """Staff self-registration -> created as 'pending' (needs admin approval)."""
    if role not in ("nurse", "doctor"):
        return False, "You can only request a nurse or doctor account."
    return create_user(username, password, role, full_name, status="pending")


def authenticate(username, password):
    username = (username or "").strip()
    row = _get_user_row(username)
    if not row:
        database.record_login(username, None, False, "no such user")
        return None
    salt = bytes.fromhex(row["salt"])
    if _hash(password, salt) != row["password_hash"]:
        database.record_login(username, row["role"], False, "bad password")
        return None
    status = row["status"] if "status" in row.keys() else "active"
    if status != "active":
        database.record_login(username, row["role"], False, "pending approval")
        return {"username": row["username"], "role": row["role"],
                "full_name": row["full_name"], "status": status}
    database.record_login(username, row["role"], True, "ok")
    return {"username": row["username"], "role": row["role"],
            "full_name": row["full_name"], "status": "active"}


def change_password(username, new_password):
    if len(new_password or "") < 6:
        return False, "Password must be at least 6 characters."
    salt = os.urandom(16)
    conn = database.get_conn()
    conn.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?",
                 (_hash(new_password, salt), salt.hex(), username))
    conn.commit()
    conn.close()
    return True, "Password updated."


def set_status(username, status):
    conn = database.get_conn()
    conn.execute("UPDATE users SET status=? WHERE username=?", (status, username))
    conn.commit()
    conn.close()


def delete_user(username):
    conn = database.get_conn()
    conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()


def list_users():
    conn = database.get_conn()
    rows = conn.execute("SELECT username,role,full_name,status,created_at FROM users "
                        "ORDER BY role,username").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_pending():
    conn = database.get_conn()
    rows = conn.execute("SELECT username,role,full_name,created_at FROM users "
                        "WHERE status='pending' ORDER BY created_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def user_count():
    conn = database.get_conn()
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return n


def seed_accounts():
    if config.SEED_DEMO_USERS and user_count() == 0:
        for u in config.DEMO_USERS:
            create_user(u["username"], u["password"], u["role"], u["full_name"])
        return
    ba = config.BOOTSTRAP_ADMIN
    if not _get_user_row(ba["username"]):
        create_user(ba["username"], ba["password"], ba["role"], ba["full_name"])
