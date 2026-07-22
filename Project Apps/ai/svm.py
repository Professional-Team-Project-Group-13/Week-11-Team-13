"""SVM-BERT baseline. Real TF-IDF + LinearSVC model with a transparent
fallback, same pattern as ai/biobert.py.

Train it first:  python train_svm.py   (needs data/train.csv, see config.py)
That saves models/svm/svm.joblib, which this module loads on first use.
If it's missing, falls back to a shrunk variant of the lexicon estimate so
the ensemble view still has a second opinion to show.
"""
import os

import config
from ai import biobert

_MODEL_PATH = os.path.join(config.MODELS_DIR, "svm", "svm.joblib")
_S = {"model": None, "tried": False, "real": False}


def _load():
    if _S["tried"]:
        return
    _S["tried"] = True
    if not os.path.exists(_MODEL_PATH):
        print(f"[svm] no trained model at {_MODEL_PATH} -> fallback estimate")
        return
    try:
        import joblib
        _S["model"] = joblib.load(_MODEL_PATH)
        _S["real"] = True
        print("[svm] live model loaded from", _MODEL_PATH)
    except Exception as exc:                       # noqa: BLE001
        print(f"[svm] load failed -> fallback estimate ({exc})")


def is_live():
    _load()
    return _S["real"]


def predict_proba(text):
    _load()
    if not _S["real"]:
        # Fallback: a slightly more conservative variant of the lexicon estimate.
        p = biobert.predict_proba(text)
        return max(0.0, min(1.0, 0.85 * p + 0.05))
    proba = _S["model"].predict_proba([text])[0]
    # class order follows what the labels looked like at training time (0/1)
    classes = list(_S["model"].classes_)
    return float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])


def engine_name():
    return "SVM (TF-IDF+LinearSVC)" if is_live() else "SVM (fallback)"
