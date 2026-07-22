"""Evaluate BioBERT and a from-scratch SVM baseline on data/test.csv, broken
down by domain (formal vs informal) -- the numbers that belong in
config.CROSS_DOMAIN.

IMPORTANT: this trains its own throw-away SVM using ONLY data/train.csv for
the evaluation numbers below. It does NOT reuse models/svm/svm.joblib,
because that file is the *production* model and (by design, see
train_svm.py) is retrained on train+test combined for best real-world
performance -- evaluating a model on data it was trained on gives inflated,
meaningless scores (data leakage). This script keeps "the model we ship"
and "the model we report numbers for" strictly separate, so the F1s below
are honest.

Run this LOCALLY where the real BioBERT weights + torch/transformers are
installed. Needs:
    - models/biobert/  (or wherever config.MODEL_DIR points) with real weights
    - data/train.csv and data/test.csv

Run from the project root:
    python eval_cross_domain.py
"""
import json

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from ai import biobert

THRESH = config.DECISION_THRESHOLD


def _load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=[config.TEXT_COL, config.LABEL_COL])
    return df


def _domain_breakdown(y_true, y_pred, domains):
    rows = []
    overall = round(f1_score(y_true, y_pred, average="macro"), 4)
    rows.append(("overall", overall))
    for dom in sorted(set(domains)):
        idx = [i for i, d in enumerate(domains) if d == dom]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        rows.append((dom, round(f1_score(yt, yp, average="macro"), 4)))
    return rows


def main():
    if not biobert.is_live():
        raise SystemExit(
            "BioBERT failed to load.（is_live()==False）。Check if config.MODEL_DIR points to the actual model directory."
            "and whether torch/transformers are properly installed."
        )

    df_train = _load_csv(config.TRAIN_CSV)
    df_test = _load_csv(config.DATA_DIR + "/test.csv")
    texts_test = df_test[config.TEXT_COL].astype(str).tolist()
    y_true = df_test[config.LABEL_COL].astype(int).tolist()
    domains = df_test[config.DOMAIN_COL].tolist() if config.DOMAIN_COL in df_test.columns else ["-"] * len(df_test)

    # ---- BioBERT: the real, already-trained model, untouched by this script ----
    print(f"[eval] {len(texts_test)} test samples; BioBERT is currently running....")
    bio_p = biobert.predict_proba_batch(texts_test)[:, 1]
    bio_pred = [int(p >= THRESH) for p in bio_p]

    # ---- SVM: trained fresh, ONLY on train.csv, purely for this evaluation ----
    print("[eval] Training a clean SVM using only train.csv (for evaluation only, not covering the production model）...")
    eval_svm = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                   sublinear_tf=True, min_df=2)),
        ("svm", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0, max_iter=5000), cv=3)),
    ])
    eval_svm.fit(df_train[config.TEXT_COL].astype(str).tolist(),
                 df_train[config.LABEL_COL].astype(int).tolist())
    svm_pred = eval_svm.predict(texts_test).tolist()
    svm_p = eval_svm.predict_proba(texts_test)
    classes = list(eval_svm.classes_)
    svm_p = [row[classes.index(1)] for row in svm_p]

    # ---- Ensemble: average of the two CLEAN probability streams above ----
    ens_p = [(b + s) / 2 for b, s in zip(bio_p, svm_p)]
    ens_pred = [int(p >= THRESH) for p in ens_p]

    results = {"BioBERT": bio_pred, "SVM": svm_pred, "Ensemble": ens_pred}
    all_rows = []
    print("\n[eval] Summary of results (all clean, no data leakage)：")
    for model_name, y_pred in results.items():
        for scope, f1 in _domain_breakdown(y_true, y_pred, domains):
            print(f"    {model_name:10s}  {scope:22s}  F1={f1}")
            all_rows.append((model_name, scope, f1))

    print("\n[eval] You can directly paste this to replace the `CROSS_DOMAIN['models']` list in `config.py`.：\n")
    print("CROSS_DOMAIN = {")
    print('    "models": [')
    for model_name, scope, f1 in all_rows:
        if scope == "overall":
            continue
        print(f'        {{"name": "{model_name}", "domain": "{scope}", "f1": {f1}}},')
    print("    ],")
    print("    # ... Retain the original overlap / sharedWords / formalOnly / informalOnly")
    print("}")

    with open(config.MODELS_DIR + "/cross_domain_eval.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n[eval] Details have been saved to {config.MODELS_DIR}/cross_domain_eval.json")
    print("\n[eval] Noted：models/svm/svm.joblib（app.py The one actually used）Has not been modified by this script."
          "remains train_svm.py The production model, fully trained on the complete dataset, is ready for use by the app.")


if __name__ == "__main__":
    main()
