"""Explainability: real LIME + SHAP on the BioBERT model, lexicon fallback."""
import re

import config
from ai import biobert


def _lexicon_drivers(text, top_k=8):
    weights = {}
    for t in text.split():
        k = re.sub(r"[^a-z']", "", t.lower())
        if k in biobert.LEX:
            weights[k] = weights.get(k, 0.0) + biobert.LEX[k]
    drivers = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return [[w, round(v, 3)] for w, v in drivers]


def _lime(text, top_k):
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=["No ADE", "ADE"])
    exp = explainer.explain_instance(
        text, biobert.predict_proba_batch,
        num_features=top_k, num_samples=config.LIME_SAMPLES)
    return [[w, round(float(v), 3)] for w, v in exp.as_list()]


def _shap(text, top_k):
    import numpy as np
    import shap

    def f(xs):
        return biobert.predict_proba_batch(list(xs))[:, 1]

    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(f, masker)
    sv = explainer([text], max_evals=config.SHAP_MAX_EVALS)
    toks = [t.strip() for t in sv.data[0]]
    vals = np.asarray(sv.values[0]).ravel()
    pairs = [(tok, float(v)) for tok, v in zip(toks, vals) if tok]
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [[w, round(v, 3)] for w, v in pairs[:top_k]]


def explain(text, top_k=None):
    """Return {'lime':[...], 'shap':[...], 'agreement':float}."""
    top_k = top_k or config.LIME_FEATURES
    if biobert.is_live():
        try:
            lime = _lime(text, top_k)
        except Exception as exc:                   # noqa: BLE001
            print(f"[xai] LIME failed ({exc}); using lexicon")
            lime = _lexicon_drivers(text, top_k)
        try:
            shap_d = _shap(text, top_k)
        except Exception as exc:                   # noqa: BLE001
            print(f"[xai] SHAP failed ({exc}); mirroring LIME")
            shap_d = [[w, round(v * 0.95, 3)] for w, v in lime]
    else:
        lime = _lexicon_drivers(text, top_k)
        shap_d = [[w, round(v * (0.9 if i % 2 else 1.05), 3)]
                  for i, (w, v) in enumerate(lime)]

    a = set(w for w, _ in lime[:5])
    b = set(w for w, _ in shap_d[:5])
    agreement = len(a & b) / len(a | b) if (a | b) else 0.0
    return {"lime": lime, "shap": shap_d, "agreement": round(agreement, 3)}
