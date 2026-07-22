"""
Central configuration.

"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "safetynet.db")

# ---------------------------------------------------------------------------
# Live inference. The app expects the REAL fine-tuned BioBERT here.
# Drop your saved model folder (config.json + tokenizer + weights) into
# models/biobert/ and this runs live. Set to None only to force preview.
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(MODELS_DIR, "biobert_model")
MAX_LEN = 128

# Training data used to build the FAISS precedent index (real retrieval).
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEXT_COL = "text"       # column with the raw text
LABEL_COL = "label"     # 0 / 1
DOMAIN_COL = "domain"   # 'formal' / 'informal' (optional)

# Explainability + retrieval parameters
LIME_SAMPLES = 200
LIME_FEATURES = 8
SHAP_MAX_EVALS = 120
FAISS_K = 3

APP_NAME = "SafetyNet AI"
APP_TAGLINE = "Cross-domain adverse-event triage"

ROLES = ["patient", "nurse", "doctor", "admin"]

# --- Accounts --------------------------------------------------------------
# Real accounts only. Demo seeding is OFF. On first run a single bootstrap
# ADMIN is created so you can log in and create the real staff/patient
# accounts. CHANGE THIS PASSWORD immediately (Admin > Users > reset password).
SEED_DEMO_USERS = False
ALLOW_PATIENT_SIGNUP = True     # patients can self-register on the login page

BOOTSTRAP_ADMIN = {"username": "admin", "password": "ChangeMe!Admin1",
                   "role": "admin", "full_name": "System Administrator"}

# Only used if SEED_DEMO_USERS = True (handy while developing)
DEMO_USERS = [
    {"username": "patient", "password": "patient123", "role": "patient", "full_name": "Demo Patient"},
    {"username": "nurse",   "password": "nurse123",   "role": "nurse",   "full_name": "Nurse Okafor"},
    {"username": "doctor",  "password": "doctor123",  "role": "doctor",  "full_name": "Dr. Bello"},
    {"username": "admin",   "password": "admin123",   "role": "admin",   "full_name": "System Admin"},
]

# Triage decision threshold (0.5 operating point) and priority bands
DECISION_THRESHOLD = 0.50
TRIAGE_BANDS = [   # (low%, high%, label, colour, priority)
    (0,  20,  "Low concern",          "#0FA36B", "P4"),
    (20, 50,  "Monitor",              "#0E7C74", "P3"),
    (50, 80,  "Elevated — review",    "#E0930C", "P2"),
    (80, 101, "High — urgent review", "#DC2626", "P1"),
]

# Red-flag symptoms that force urgent escalation regardless of model score
RED_FLAGS = [
    "difficulty breathing", "trouble breathing", "shortness of breath",
    "chest pain", "swelling of the face", "swollen throat", "throat swelling",
    "anaphylaxis", "collapsed", "unconscious", "seizure", "severe bleeding",
]

# Vital-sign alert thresholds (very simplified, illustrative)
VITALS_ALERTS = {
    "heart_rate":   {"low": 50,  "high": 120},
    "systolic_bp":  {"low": 90,  "high": 180},
    "temperature":  {"low": 35.0, "high": 38.5},
    "spo2":         {"low": 92,  "high": 101},
}

# ---------------------------------------------------------------------------
# Cross-domain results  
# ---------------------------------------------------------------------------
CROSS_DOMAIN = {
    "models": [
        {"name": "BioBERT", "domain": "Formal (DailyMed)", "f1": 0.9959},
        {"name": "BioBERT", "domain": "Informal (CADEC)", "f1": 0.85},
        {"name": "SVM", "domain": "Formal (DailyMed)", "f1": 0.9959},
        {"name": "SVM", "domain": "Informal (CADEC)", "f1": 0.7505},
        {"name": "Ensemble", "domain": "Formal (DailyMed)", "f1": 0.9959},
        {"name": "Ensemble", "domain": "Informal (CADEC)", "f1": 0.8664},
    ],
    "overlap": 0.24,
    "sharedWords": ["rash", "nausea", "dizziness"],
    "formalOnly": ["administration", "discontinued", "hypersensitivity"],
    "informalOnly": ["knocked me out", "couldn't", "woozy"],
}
