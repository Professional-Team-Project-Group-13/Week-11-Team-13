"""FAISS retrieval of similar known cases in BioBERT embedding space.

Builds an index once from config.TRAIN_CSV using BioBERT embeddings. Falls back
to curated example precedents if the model / data / faiss are unavailable.
"""
import os

import config
from ai import biobert

_FALLBACK = {
    "formal": [
        {"text": "Patients experienced hypersensitivity reactions including rash and pruritus after dosing.", "sim": 0.90, "label": 1, "domain": "Formal"},
        {"text": "Nausea and vomiting were the most frequently reported adverse reactions.", "sim": 0.86, "label": 1, "domain": "Formal"},
        {"text": "No clinically significant adverse events were observed in the control arm.", "sim": 0.63, "label": 0, "domain": "Formal"},
    ],
    "informal": [
        {"text": "felt really dizzy and nauseous all day after taking it, ended up in bed", "sim": 0.84, "label": 1, "domain": "Informal"},
        {"text": "came out in a red itchy rash on my arms within hours of the first dose", "sim": 0.80, "label": 1, "domain": "Informal"},
        {"text": "been on these for months now and feel totally fine, no issues at all", "sim": 0.55, "label": 0, "domain": "Informal"},
    ],
}

_IDX = {"built": False, "index": None, "meta": None}


def _build():
    if _IDX["built"]:
        return
    _IDX["built"] = True
    if not (biobert.is_live() and os.path.exists(config.TRAIN_CSV)):
        return
    try:
        import faiss
        import pandas as pd
        df = pd.read_csv(config.TRAIN_CSV)
        text_col = config.TEXT_COL if config.TEXT_COL in df.columns else df.columns[0]
        texts = df[text_col].astype(str).tolist()
        # cap for a snappy first build; raise if you want the full corpus
        texts = texts[:4000]
        emb = biobert.embed(texts)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        meta = []
        for i in range(len(texts)):
            lab = int(df[config.LABEL_COL].iloc[i]) if config.LABEL_COL in df.columns else 0
            dom = str(df[config.DOMAIN_COL].iloc[i]) if config.DOMAIN_COL in df.columns else "-"
            meta.append({"text": texts[i], "label": lab, "domain": dom.title()})
        _IDX["index"], _IDX["meta"] = index, meta
        print(f"[faiss] index built with {len(texts)} cases")
    except Exception as exc:                       # noqa: BLE001
        print(f"[faiss] build failed -> fallback precedents ({exc})")


def retrieve(text, k=None):
    k = k or config.FAISS_K
    _build()
    if _IDX["index"] is not None:
        q = biobert.embed([text])
        sims, ids = _IDX["index"].search(q, k)
        out = []
        for sim, idx in zip(sims[0], ids[0]):
            if idx < 0:
                continue
            m = dict(_IDX["meta"][idx])
            m["sim"] = round(float(sim), 3)
            out.append(m)
        if out:
            return out
    return _FALLBACK.get(biobert.guess_domain(text), _FALLBACK["formal"])[:k]
