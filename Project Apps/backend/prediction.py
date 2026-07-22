"""Runs the full AI pipeline for a piece of text and returns one dict."""
from ai import biobert, explainability, faiss_engine, risk_model, svm


def analyze(text):
    proba = biobert.predict_proba(text)
    svm_p = svm.predict_proba(text)
    ensemble = round((proba + svm_p) / 2, 3)
    xai = explainability.explain(text)
    return {
        "proba": round(proba, 3),
        "svm_proba": round(svm_p, 3),
        "ensemble_proba": ensemble,
        "severity": risk_model.severity_from_text(text),
        "domain": biobert.guess_domain(text),
        "engine": "biobert" if biobert.is_live() else "preview",
        "lime": xai["lime"],
        "shap": xai["shap"],
        "agreement": xai["agreement"],
        "neighbours": faiss_engine.retrieve(text, k=3),
        "red_flags": risk_model.red_flags(text),
    }
