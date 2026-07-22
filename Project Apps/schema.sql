-- =====================================================================
-- Database schema (SQLite)
-- The application (backend/database.py) creates this automatically on
-- first run as `safetynet.db`. This file documents the schema and lets
-- you (re)create it manually:   sqlite3 safetynet.db < schema.sql
-- Passwords are PBKDF2-hashed by the app; never insert plaintext here.
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---- users ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,           -- PBKDF2-HMAC-SHA256 (hex)
    salt          TEXT NOT NULL,           -- per-user random salt (hex)
    role          TEXT NOT NULL CHECK (role IN ('patient','nurse','doctor','admin')),
    full_name     TEXT,
    status        TEXT DEFAULT 'active',    -- 'active' or 'pending' (staff awaiting approval)
    created_at    TEXT
);

-- ---- cases (patient submissions + triage + clinician decision) -------
CREATE TABLE IF NOT EXISTS cases (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_username   TEXT,
    created_at         TEXT,
    symptoms           TEXT,
    medication         TEXT,
    age                INTEGER,
    heart_rate         REAL,
    systolic_bp        REAL,
    temperature        REAL,
    spo2               REAL,
    proba              REAL,               -- P(ADE) from BioBERT
    label              INTEGER,            -- 0 / 1
    severity           TEXT,
    triage_band        TEXT,
    priority           TEXT,               -- P1..P4
    action             TEXT,
    domain             TEXT,               -- formal / informal
    engine             TEXT,               -- biobert / preview
    red_flag           INTEGER DEFAULT 0,
    status             TEXT DEFAULT 'pending',  -- pending / escalated / closed
    clinician_username TEXT,
    clinician_note     TEXT,
    decided_at         TEXT,
    analysis_json      TEXT                -- full LIME/SHAP/FAISS payload
);

-- ---- audit_log (every meaningful action) ----------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT,
    username TEXT,
    role     TEXT,
    action   TEXT,
    case_id  INTEGER,
    detail   TEXT
);

-- ---- login_records (successful AND failed sign-in attempts) ---------
CREATE TABLE IF NOT EXISTS login_records (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT,
    username TEXT,
    role     TEXT,
    success  INTEGER,                      -- 1 = success, 0 = failed
    detail   TEXT
);

CREATE INDEX IF NOT EXISTS idx_cases_status  ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_patient ON cases(patient_username);
CREATE INDEX IF NOT EXISTS idx_login_user    ON login_records(username);
CREATE INDEX IF NOT EXISTS idx_audit_user    ON audit_log(username);
