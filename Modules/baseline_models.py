"""Baseline ADE classifiers for SafetyNet AI (Task 5).

Three classical text classifiers - Logistic Regression, Random Forest, and
XGBoost - over a shared TF-IDF representation. Each model is a scikit-learn
Pipeline so the vectoriser is fit on the TRAINING split only, which makes
data leakage structurally impossible.

"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import eval_utils as ev

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover - environment without xgboost
    _HAS_XGB = False

logger = logging.getLogger("baseline_models")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

RANDOM_SEED = 42
EXPECTED_COLS = (ev.TEXT_COL, ev.LABEL_COL)


def validate_columns(df: pd.DataFrame,
                     cols: Tuple[str, ...] = EXPECTED_COLS) -> None:
    """Fail fast if a required column is missing or the labels are not binary.

    Validating inputs before any heavy compute is both a correctness and a
    security practice (no silent misbehaviour on malformed frames).
    """
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col!r}")
    bad = set(pd.unique(df[ev.LABEL_COL])) - {0, 1}
    if bad:
        raise ValueError(f"Labels must be binary 0/1; found extra: {bad}")


def build_vectorizer(max_features: int = 20000,
                     ngram_range: Tuple[int, int] = (1, 2),
                     min_df: int = 2) -> TfidfVectorizer:
    """Return a TF-IDF vectoriser with project defaults (word + bigram)."""
    return TfidfVectorizer(max_features=max_features,
                           ngram_range=ngram_range, min_df=min_df,
                           sublinear_tf=True)


def make_models(seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Return the baseline model zoo (single source of truth).

    Each classifier handles class imbalance (LR/RF via ``class_weight``,
    XGBoost via ``scale_pos_weight`` set later from the training split).
    XGBoost is included only if the package is importable.
    """
    models: Dict[str, Any] = {
        "LogReg": LogisticRegression(max_iter=1000,
                                     class_weight="balanced",
                                     random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=300,
                                               class_weight="balanced",
                                               n_jobs=-1, random_state=seed),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=-1)
    else:
        logger.warning("xgboost not installed - skipping XGBoost baseline.")
    return models


def _scale_pos_weight(y: np.ndarray) -> float:
    """Ratio negatives/positives, used to balance XGBoost (1.0 if no pos)."""
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return neg / pos if pos else 1.0


def make_pipeline(classifier: Any,
                  vectorizer: Optional[TfidfVectorizer] = None) -> Pipeline:
    """Wrap a classifier with TF-IDF so the vectoriser fits on train only.

    Using a Pipeline is the structural guarantee against leakage: calling
    ``.fit(train)`` fits the vectoriser on the training text exclusively.
    """
    return Pipeline([("tfidf", vectorizer or build_vectorizer()),
                     ("clf", classifier)])


def train_and_predict(classifier: Any, train_df: pd.DataFrame,
                      eval_df: pd.DataFrame) -> Tuple[Pipeline, np.ndarray]:
    """Fit a baseline on train_df and return (pipeline, P(ADE) on eval_df)."""
    validate_columns(train_df)
    validate_columns(eval_df)
    if hasattr(classifier, "scale_pos_weight"):
        classifier.set_params(
            scale_pos_weight=_scale_pos_weight(train_df[ev.LABEL_COL].values))
    pipe = make_pipeline(classifier)
    pipe.fit(train_df[ev.TEXT_COL].fillna(""), train_df[ev.LABEL_COL])
    proba = pipe.predict_proba(eval_df[ev.TEXT_COL].fillna(""))[:, 1]
    return pipe, proba


def evaluate_all(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 val_df: Optional[pd.DataFrame] = None,
                 seed: int = RANDOM_SEED
                 ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Train every baseline and evaluate on the test split.

    Returns a tidy metrics DataFrame plus a per-model dict holding the fitted
    pipeline and the test/val probabilities (for the ensemble handoff).
    """
    rows: List[Dict[str, Any]] = []
    artefacts: Dict[str, Dict[str, Any]] = {}
    for name, clf in make_models(seed).items():
        logger.info("Training baseline: %s", name)
        pipe, test_proba = train_and_predict(clf, train_df, test_df)
        test_pred = (test_proba >= 0.5).astype(int)
        metrics = ev.compute_metrics(test_df[ev.LABEL_COL].values,
                                     test_pred, test_proba)
        rows.append({"model": name, **metrics})
        artefacts[name] = {"pipeline": pipe, "test_proba": test_proba}
        if val_df is not None:
            artefacts[name]["val_proba"] = pipe.predict_proba(
                val_df[ev.TEXT_COL].fillna(""))[:, 1]
    return ev.results_frame(rows), artefacts


def top_features(pipe: Pipeline, n: int = 15
                 ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Return the top-n tokens pushing toward ADE and toward no-ADE.

    For Logistic Regression these are signed coefficients; for tree models
    they are (always positive) importances, returned as the ADE-side list.
    """
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
        order = np.argsort(weights)
        neg = [(vocab[i], float(weights[i])) for i in order[:n]]
        pos = [(vocab[i], float(weights[i])) for i in order[::-1][:n]]
        return pos, neg
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return [], []
    order = np.argsort(importances)[::-1][:n]
    pos = [(vocab[i], float(importances[i])) for i in order]
    return pos, []


def run_cross_domain(full_df: pd.DataFrame, classifier_name: str,
                     train_domain: str = "Formal",
                     test_domain: str = "Informal",
                     seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Train one baseline on a domain, test on another; return metrics."""
    train_df = ev.split_by_domain(full_df, train_domain)
    test_df = ev.split_by_domain(full_df, test_domain)
    clf = make_models(seed)[classifier_name]
    _, proba = train_and_predict(clf, train_df, test_df)
    pred = (proba >= 0.5).astype(int)
    return ev.compute_metrics(test_df[ev.LABEL_COL].values, pred, proba)
