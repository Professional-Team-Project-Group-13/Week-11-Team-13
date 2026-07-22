"""
 Shared evaluation utilities.

Common metric computation, result formatting, and cross-domain helpers reused
by every model tier (baselines, Bi-LSTM, SVM-BERT, BioBERT, ensemble) so that
results are directly comparable and the metric definitions live in one place.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

TEXT_COL = "text_processed"
RAW_TEXT_COL = "text"
LABEL_COL = "label"
DOMAIN_COL = "domain"
RANDOM_STATE = 42


def compute_metrics(y_true, y_pred, y_proba=None):
    """Return the project's standard metric dictionary for binary ADE labels."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def split_by_domain(df, domain_substring):
    """Return the rows whose domain column contains the given substring."""
    return df[df[DOMAIN_COL].str.contains(domain_substring, case=False)]


def degradation_row(model_name, f1_standard, f1_cross):
    """Return a dict summarising the cross-domain generalisation gap."""
    if f1_standard and not np.isnan(f1_standard) and f1_standard != 0:
        deg = (f1_standard - f1_cross) / f1_standard * 100
    else:
        deg = float("nan")
    return {
        "model": model_name,
        "f1_standard": round(f1_standard, 4),
        "f1_cross": round(f1_cross, 4),
        "degradation_pct": round(deg, 1),
    }


def results_frame(rows):
    """Build a sorted results DataFrame from a list of metric dicts."""
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(
        drop=True
    )


def load_splits(train_path, val_path, test_path, full_path=None):
    """Load the preprocessed CSV splits produced by the data pipeline."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    if full_path:
        full_df = pd.read_csv(full_path)
    else:
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return train_df, val_df, test_df, full_df
