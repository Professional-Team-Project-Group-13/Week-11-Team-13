"""Train the SVM baseline for SafetyNet AI.

Trains a TF-IDF + LinearSVC classifier (with probability calibration so it
exposes predict_proba, same as BioBERT does) on config.TRAIN_CSV, evaluates
it on a real held-out test set (data/test.csv, same columns), and reports
both overall and per-domain (formal vs informal) F1 -- the numbers that
belong in config.CROSS_DOMAIN.

Run from the project root:
    python train_svm.py

Requires data/train.csv (and ideally data/test.csv) with the columns named
in config.py: TEXT_COL="text", LABEL_COL="label", DOMAIN_COL="domain"
(optional but needed for the per-domain breakdown).
"""
import json
import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

import config

OUT_DIR = os.path.join(config.MODELS_DIR, "svm")
OUT_PATH = os.path.join(OUT_DIR, "svm.joblib")
TEST_CSV = os.path.join(config.DATA_DIR, "test.csv")
EVAL_OUT = os.path.join(OUT_DIR, "eval_results.json")


def _load_csv(path):
    df = pd.read_csv(path)
    if config.TEXT_COL not in df.columns or config.LABEL_COL not in df.columns:
        raise SystemExit(
            f"{path} 里缺少必要列，需要 '{config.TEXT_COL}' 和 '{config.LABEL_COL}'，"
            f"当前列为: {list(df.columns)}"
        )
    df = df.dropna(subset=[config.TEXT_COL, config.LABEL_COL])
    return df


def _domain_f1(pipeline, df):
    """Per-domain macro-F1, using whatever values are in the domain column."""
    if config.DOMAIN_COL not in df.columns:
        return {}
    out = {}
    for dom, sub in df.groupby(config.DOMAIN_COL):
        y_true = sub[config.LABEL_COL].astype(int).tolist()
        y_pred = pipeline.predict(sub[config.TEXT_COL].astype(str).tolist())
        out[dom] = round(f1_score(y_true, y_pred, average="macro"), 4)
    return out


def main():
    if not os.path.exists(config.TRAIN_CSV):
        raise SystemExit(
            f"未找到训练数据：{config.TRAIN_CSV}\n"
            f"请把带有 '{config.TEXT_COL}' 和 '{config.LABEL_COL}' 两列的 csv 放到这个路径。"
        )

    df_train = _load_csv(config.TRAIN_CSV)
    X_train_all = df_train[config.TEXT_COL].astype(str).tolist()
    y_train_all = df_train[config.LABEL_COL].astype(int).tolist()
    print(f"[train_svm] 训练集 {len(X_train_all)} 条，ADE 占比 {sum(y_train_all)/len(y_train_all):.1%}")

    # 真实测试集（推荐）：如果 data/test.csv 存在就用它做最终评估，
    # 不存在则从训练集里切一份 20% 内部测试集。
    if os.path.exists(TEST_CSV):
        df_test = _load_csv(TEST_CSV)
        print(f"[train_svm] 使用真实测试集 {TEST_CSV}，{len(df_test)} 条")
        X_fit, y_fit = X_train_all, y_train_all
    else:
        print("[train_svm] 未找到 data/test.csv，从训练集内部切 20% 做测试")
        X_fit, X_hold, y_fit, y_hold = train_test_split(
            X_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
        )
        df_test = pd.DataFrame({config.TEXT_COL: X_hold, config.LABEL_COL: y_hold})

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("svm", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0, max_iter=5000),
            cv=3,
        )),
    ])

    print("[train_svm] 训练中 ...")
    pipeline.fit(X_fit, y_fit)

    y_test = df_test[config.LABEL_COL].astype(int).tolist()
    y_pred = pipeline.predict(df_test[config.TEXT_COL].astype(str).tolist())
    print("[train_svm] 测试集整体表现：")
    print(classification_report(y_test, y_pred, target_names=["No ADE", "ADE"]))
    overall_f1 = round(f1_score(y_test, y_pred, average="macro"), 4)
    print(f"[train_svm] 整体 macro F1 = {overall_f1}")

    domain_f1 = _domain_f1(pipeline, df_test)
    if domain_f1:
        print("[train_svm] 分领域 F1（可直接填进 config.CROSS_DOMAIN）：")
        for dom, f1 in domain_f1.items():
            print(f"    {dom}: {f1}")

    # 最终交付模型：如果用了独立测试集，再用全部数据（train+test）重训一次；
    # 如果是内部切分，就用全量训练集重训。
    print("[train_svm] 用全部可用数据重训最终交付模型 ...")
    if os.path.exists(TEST_CSV):
        df_all = pd.concat([df_train, df_test], ignore_index=True)
    else:
        df_all = df_train
    pipeline.fit(df_all[config.TEXT_COL].astype(str).tolist(),
                 df_all[config.LABEL_COL].astype(int).tolist())

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(pipeline, OUT_PATH)
    print(f"[train_svm] 模型已保存到 {OUT_PATH}")

    with open(EVAL_OUT, "w") as f:
        json.dump({"overall_macro_f1": overall_f1, "domain_macro_f1": domain_f1}, f, indent=2)
    print(f"[train_svm] 评估结果已保存到 {EVAL_OUT}")


if __name__ == "__main__":
    main()
