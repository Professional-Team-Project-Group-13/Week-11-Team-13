"""SQLite persistence: users, cases, and an audit log."""
import os
import sqlite3
from datetime import datetime

import config


def get_conn():
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    for d in (config.DATA_DIR, config.MODELS_DIR, config.REPORTS_DIR, config.UPLOADS_DIR):
        os.makedirs(d, exist_ok=True)
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL,
        full_name TEXT,
        status TEXT DEFAULT 'active',
        created_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS cases(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_username TEXT,
        created_at TEXT,
        symptoms TEXT,
        medication TEXT,
        age INTEGER,
        heart_rate REAL, systolic_bp REAL, temperature REAL, spo2 REAL,
        proba REAL, label INTEGER, severity TEXT,
        triage_band TEXT, priority TEXT, action TEXT, domain TEXT, engine TEXT,
        red_flag INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        clinician_username TEXT, clinician_note TEXT, decided_at TEXT,
        analysis_json TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, username TEXT, role TEXT, action TEXT,
        case_id INTEGER, detail TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS login_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, username TEXT, role TEXT,
        success INTEGER, detail TEXT)""")
    # migration: ensure users.status exists on older databases
    cols = [r[1] for r in c.execute("PRAGMA table_info(users)").fetchall()]
    if "status" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'")
    conn.commit()
    conn.close()


# ---- audit -----------------------------------------------------------------
def log_action(username, role, action, case_id=None, detail=""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO audit_log(ts,username,role,action,case_id,detail) VALUES(?,?,?,?,?,?)",
        (datetime.utcnow().isoformat(timespec="seconds"), username, role, action, case_id, detail))
    conn.commit()
    conn.close()


def recent_audit(limit=100):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def record_login(username, role, success, detail=""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO login_records(ts,username,role,success,detail) VALUES(?,?,?,?,?)",
        (datetime.utcnow().isoformat(timespec="seconds"), username,
         role or "-", 1 if success else 0, detail))
    conn.commit()
    conn.close()


def recent_logins(limit=100):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM login_records ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- cases -----------------------------------------------------------------
def create_case(data):
    conn = get_conn()
    cols = ",".join(data.keys())
    ph = ",".join("?" for _ in data)
    cur = conn.execute(f"INSERT INTO cases({cols}) VALUES({ph})", list(data.values()))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def update_case(case_id, **fields):
    if not fields:
        return
    conn = get_conn()
    sets = ",".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE cases SET {sets} WHERE id=?", list(fields.values()) + [case_id])
    conn.commit()
    conn.close()


def get_case(case_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_cases(status=None, patient_username=None, limit=500):
    conn = get_conn()
    q, params = "SELECT * FROM cases", []
    where = []
    if status:
        where.append("status=?"); params.append(status)
    if patient_username:
        where.append("patient_username=?"); params.append(patient_username)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY (priority='P1') DESC, id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def case_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    ade = conn.execute("SELECT COUNT(*) FROM cases WHERE label=1").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM cases WHERE status='pending'").fetchone()[0]
    escalated = conn.execute("SELECT COUNT(*) FROM cases WHERE status='escalated'").fetchone()[0]
    closed = conn.execute("SELECT COUNT(*) FROM cases WHERE status='closed'").fetchone()[0]
    p1 = conn.execute("SELECT COUNT(*) FROM cases WHERE priority='P1'").fetchone()[0]
    conn.close()
    return {"total": total, "ade": ade, "pending": pending,
            "escalated": escalated, "closed": closed, "p1": p1,
            "referral_rate": (ade / total * 100) if total else 0.0}
