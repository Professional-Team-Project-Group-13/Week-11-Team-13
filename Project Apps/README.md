# SafetyNet AI — Clinical Triage Console (multi-portal)

A role-based clinical-triage application for adverse drug event (ADE) detection
across formal (DailyMed) and informal (CADEC) text. Built in Streamlit around the
`healthcare_ai/` architecture, with a patient self-intake portal, nurse and doctor
dashboards, an admin/analytics portal, explainable-AI reasoning, FAISS retrieval,
a clinical triage engine, PDF reports, and a privacy-checklist gate.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Complete the privacy checklist, then sign in.

### Demo accounts
| role | username | password |
|------|----------|----------|
| Patient | patient | patient123 |
| Nurse   | nurse   | nurse123 |
| Doctor  | doctor  | doctor123 |
| Admin   | admin   | admin123 |

## Flow
Patient submits symptoms + medication → BioBERT ADE prediction → clinical triage
engine (severity, vitals, age, red flags) → case enters the nurse queue →
nurse reviews explainability (LIME/SHAP), FAISS precedents, and either closes or
escalates to a doctor → doctor makes the final decision and can export a PDF
report. Admin sees analytics, cross-domain findings, users, and the audit log.

## Live BioBERT (this app is wired for the real model)
The app is configured to run the **real fine-tuned BioBERT** by default:
`MODEL_DIR` points at `models/biobert/`.

1. Put your saved model folder in `models/biobert/` — it must contain
   `config.json`, the tokenizer files, and the weights
   (`model.safetensors` or `pytorch_model.bin`). Do NOT commit it (add `models/`
   to `.gitignore`; the 435 MB weights exceed GitHub's limit).
2. Install the stack: `pip install -r requirements.txt` (includes transformers,
   torch, faiss-cpu, lime, shap).
3. For real FAISS precedents, put `train.csv` in `data/` and set the column names
   in `config.py` (`TEXT_COL`, `LABEL_COL`, `DOMAIN_COL`).

The top of every portal shows a banner: green **BioBERT live** when the model
loaded, amber **preview** otherwise. If the model or a library is missing, that
component falls back transparently so the app never crashes — but predictions are
only real when the banner is green.

Replace the numbers in `config.CROSS_DOMAIN` with your real `eval_utils` results.

## Theme
Dark clinical-console theme with translucent panels. Enforced via
`.streamlit/config.toml` (`base="dark"`) plus `frontend/theme.py`, so text stays
readable regardless of the viewer's system setting.

## Architecture
```
healthcare_ai/
├── app.py                 entry: privacy gate, auth, role routing
├── config.py              paths, thresholds, cross-domain metrics, demo users  <-- edit
├── requirements.txt
├── backend/
│   ├── auth.py            PBKDF2 auth, 4 roles, seeding
│   ├── database.py        SQLite: users, cases, audit
│   ├── prediction.py      orchestrates the AI layer
│   ├── triage.py          clinical triage engine
│   └── reports.py         PDF report generation
├── ai/
│   ├── biobert.py         ADE classifier (+ lexicon fallback)
│   ├── svm.py             baseline
│   ├── faiss_engine.py    similar-case retrieval
│   ├── explainability.py  LIME/SHAP drivers
│   └── risk_model.py      severity, vitals, red flags
├── frontend/
│   ├── login.py  patient.py  nurse.py  doctor.py  admin.py
│   ├── theme.py           CSS + HTML components (gauge, bars, precedents)
│   └── casedetail.py      shared clinician review panel
├── models/  data/  uploads/  reports/
```

**Not a medical device.** Decision support only; a qualified clinician makes the
final call. Do not enter patient-identifiable data.


## Accounts, logins & database
This app uses a **real SQLite database** (`safetynet.db`), created automatically
on first run. There are **no demo accounts**.

- On first run a single **bootstrap admin** is created from `config.BOOTSTRAP_ADMIN`
  (default `admin` / `ChangeMe!Admin1`). **Sign in and change this password immediately**
  (Admin → Accounts → Reset password).
- The **admin** creates staff accounts (nurse / doctor / admin) under
  Admin → Accounts → Create account.
- **Patients** self-register on the login page (toggle with `ALLOW_PATIENT_SIGNUP`).
- Every sign-in — **successful and failed** — is stored in `login_records` and shown
  under Admin → Login records. All actions are also written to `audit_log`.

Passwords are hashed with PBKDF2-HMAC-SHA256 + a per-user salt; plaintext is never stored.

### SQL schema
`schema.sql` (SQLite) and `schema_postgres.sql` (PostgreSQL) document every table
(`users`, `cases`, `audit_log`, `login_records`). The app builds the SQLite schema
itself; you can also recreate it manually:
```bash
sqlite3 safetynet.db < schema.sql
```
Want a Postgres server instead of the file DB? Use `schema_postgres.sql` and ask me
for the `psycopg` adapter for `backend/database.py`.


## Access model (who sees what)
- **Login is split into two entrances:** a **Patient** entrance (with self-registration)
  and a **Clinician / Staff** entrance (nurse · doctor · admin). Signing in on the wrong
  entrance is refused, and routing sends each account only to its own portal — a patient
  can never reach a staff page.
- **Patient** — submits symptoms/medication, gets a triage result.
- **Nurse** — lightweight **confirm or escalate** (prediction + triage only; no explainability).
- **Doctor / clinician** — full **clinical review**: LIME/SHAP explainability, FAISS
  precedents, final decision, and PDF report.
- **Admin** — accounts, login records, analytics, audit log.
