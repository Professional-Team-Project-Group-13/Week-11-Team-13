"""BioBERT ADE detector — real fine-tuned model with a transparent fallback.

Loads the model from config.MODEL_DIR. Exposes:
  is_live()                  -> True if the real model loaded
  predict_proba(text)        -> P(ADE) float
  predict_proba_batch(texts) -> np.ndarray (n, 2)  [P(No ADE), P(ADE)]  (for LIME/SHAP)
  embed(texts)               -> np.ndarray (n, dim) mean-pooled embeddings (for FAISS)
"""
import math
import os
import re

import config

# fallback lexicon (only used when the real model is not present)
LEX = {
    "rash": .9, "nausea": .85, "nauseous": .85, "dizzy": .8, "dizziness": .8,
    "vomiting": .85, "sick": .7, "headache": .6, "headaches": .6, "swelling": .9,
    "itchy": .7, "itching": .7, "woozy": .75, "drowsy": .6, "breathing": .95,
    "collapsed": .95, "hospital": .9, "emergency": .95, "severe": .7, "pain": .6,
    "cramps": .6, "diarrhea": .7, "diarrhoea": .7, "bleeding": .85, "chest": .7,
    "palpitations": .8, "reaction": .7, "reactions": .7, "allergic": .85,
    "hives": .85, "fever": .6, "fatigue": .5, "insomnia": .55, "knocked": .5,
    "unable": .6, "couldnt": .5, "stopped": .5, "discontinued": .6, "adverse": .6,
    "throat": .7, "seizure": .9,
    "fine": -.7, "normal": -.6, "no": -.5, "none": -.6, "good": -.5, "better": -.55,
    "relief": -.6, "tolerated": -.7, "well": -.5, "coincidence": -.6,
    "unrelated": -.7, "improved": -.6, "ok": -.5, "okay": -.5,
}

_INFORMAL = re.compile(r"\b(i|my|me|felt|couldn't|honestly|sofa|really|lol)\b", re.I)

_S = {"model": None, "tok": None, "real": False, "tried": False, "torch": None}


def _load():
    if _S["tried"]:
        return
    _S["tried"] = True
    md = config.MODEL_DIR
    if not md or not os.path.isdir(md):
        print(f"[biobert] no model at {md} -> preview mode")
        return
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)
        _S["tok"] = AutoTokenizer.from_pretrained(md)
        _S["model"] = AutoModelForSequenceClassification.from_pretrained(md)
        _S["model"].eval()
        _S["torch"] = torch
        _S["real"] = True
        print("[biobert] live model loaded from", md)
    except Exception as exc:                       # noqa: BLE001
        print(f"[biobert] load failed -> preview mode ({exc})")


def is_live():
    _load()
    return _S["real"]


def _lex_p(text):
    total = sum(LEX.get(re.sub(r"[^a-z']", "", t.lower()), 0.0) for t in text.split())
    return 1.0 / (1.0 + math.exp(-1.15 * total))


def predict_proba(text):
    _load()
    if not _S["real"]:
        return _lex_p(text)
    return float(predict_proba_batch([text])[0, 1])


def predict_proba_batch(texts):
    """Return (n,2) probabilities. Works for LIME/SHAP callbacks."""
    import numpy as np
    _load()
    if not _S["real"]:
        p = np.array([_lex_p(t) for t in texts])
        return np.vstack([1 - p, p]).T
    torch = _S["torch"]
    out = []
    bs = 16
    for i in range(0, len(texts), bs):
        batch = list(texts[i:i + bs])
        enc = _S["tok"](batch, truncation=True, max_length=config.MAX_LEN,
                        padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = _S["model"](**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out.append(probs)
    return np.vstack(out)


def embed(texts):
    """Mean-pooled last-hidden-state embeddings for FAISS. Returns (n, dim)."""
    import numpy as np
    _load()
    if not _S["real"]:
        # deterministic hash-based pseudo-embedding so preview FAISS still works
        dim = 64
        vecs = []
        for t in texts:
            v = np.zeros(dim)
            for j, tok in enumerate(re.findall(r"[a-z]+", t.lower())):
                v[hash(tok) % dim] += 1.0
            n = np.linalg.norm(v) or 1.0
            vecs.append(v / n)
        return np.vstack(vecs).astype("float32")
    torch = _S["torch"]
    out = []
    bs = 16
    for i in range(0, len(texts), bs):
        batch = list(texts[i:i + bs])
        enc = _S["tok"](batch, truncation=True, max_length=config.MAX_LEN,
                        padding=True, return_tensors="pt")
        with torch.no_grad():
            hs = _S["model"](**enc, output_hidden_states=True).hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hs * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            emb = (summed / counts).cpu().numpy()
        out.append(emb)
    arr = np.vstack(out).astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def guess_domain(text):
    return "informal" if _INFORMAL.search(text) else "formal"


def engine_name():
    return "BioBERT" if is_live() else "Preview"
