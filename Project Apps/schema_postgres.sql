-- =====================================================================
-- SafetyNet AI — database schema (PostgreSQL)
-- Optional: use this if you deploy with a Postgres server instead of the
-- built-in SQLite file. Create a database, then:
--   psql -d safetynet -f schema_postgres.sql
-- To actually connect the app to Postgres you would swap backend/database.py
-- to use psycopg (ask and I'll provide that adapter).
-- =====================================================================

CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('patient','nurse','doctor','admin')),
    full_name     TEXT,
    status        TEXT DEFAULT 'active',
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cases (
    id                 BIGSERIAL PRIMARY KEY,
    patient_username   TEXT REFERENCES users(username) ON DELETE SET NULL,
    created_at         TIMESTAMPTZ DEFAULT now(),
    symptoms           TEXT,
    medication         TEXT,
    age                INTEGER,
    heart_rate         REAL,
    systolic_bp        REAL,
    temperature        REAL,
    spo2               REAL,
    proba              REAL,
    label              INTEGER,
    severity           TEXT,
    triage_band        TEXT,
    priority           TEXT,
    action             TEXT,
    domain             TEXT,
    engine             TEXT,
    red_flag           INTEGER DEFAULT 0,
    status             TEXT DEFAULT 'pending',
    clinician_username TEXT REFERENCES users(username) ON DELETE SET NULL,
    clinician_note     TEXT,
    decided_at         TIMESTAMPTZ,
    analysis_json      JSONB
);

CREATE TABLE IF NOT EXISTS audit_log (
    id       BIGSERIAL PRIMARY KEY,
    ts       TIMESTAMPTZ DEFAULT now(),
    username TEXT,
    role     TEXT,
    action   TEXT,
    case_id  BIGINT,
    detail   TEXT
);

CREATE TABLE IF NOT EXISTS login_records (
    id       BIGSERIAL PRIMARY KEY,
    ts       TIMESTAMPTZ DEFAULT now(),
    username TEXT,
    role     TEXT,
    success  BOOLEAN,
    detail   TEXT
);

CREATE INDEX IF NOT EXISTS idx_cases_status  ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_patient ON cases(patient_username);
CREATE INDEX IF NOT EXISTS idx_login_user    ON login_records(username);
CREATE INDEX IF NOT EXISTS idx_audit_user    ON audit_log(username);
