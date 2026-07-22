"""
Ensemble Meta-Learner (Objective 2 integrator).

Combines the predicted probabilities of the three advanced SafetyNet models —
Bi-LSTM (with attention), SVM-BERT, and BioBERT — into a single stronger
prediction. Two strategies are provided:

* **Stacking** — a logistic-regression meta-learner trained on the base models'
  validation-fold probabilities, then evaluated on the held-out test fold. The
  val -> test split keeps the meta-learner honest: it never sees test labels
  during training, so the reported test metrics are leakage-free.
* **Soft voting** — a simple (optionally weighted) average of base
  probabilities. It needs no training and serves as a robust baseline to show
  that the learned stack actually adds value.

The module is deliberately model-agnostic: it consumes matrices of base-model
probabilities rather than the models themselves, so it works regardless of how
each base prediction was produced. It reads the project's standard probability
exports (``probs_<model>_val.csv`` / ``probs_<model>_test.csv``), each shaped
``uid, label, <model_name>`` and merged on ``uid`` — exactly what
``eval_utils.save_probabilities`` writes from every model notebook.
"""

import argparse
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import eval_utils as ev

logger = logging.getLogger(__name__)

# Single source of truth: the three advanced models, in canonical column order.
BASE_MODELS: Tuple[str, ...] = ("bilstm", "svm_bert", "biobert")

ArrayLike = Sequence[float]
ProbDict = Dict[str, ArrayLike]


def stack_probabilities(prob_dict: ProbDict) -> Tuple[np.ndarray, List[str]]:
    """
    Stack per-model probability vectors into a feature matrix.

    Parameters
    ----------
    prob_dict : dict[str, array-like]
        ``{model_name: probability_of_positive_class}``. All arrays must align
        in length and row order.

    Returns
    -------
    (matrix, names)
        ``matrix`` has shape ``(n_samples, n_models)``; ``names`` is the model
        order used to build its columns.
    """
    if not prob_dict:
        raise ValueError(
            "prob_dict is empty — provide at least one model's probabilities."
        )
    names = list(prob_dict)
    lengths = {len(np.asarray(prob_dict[n])) for n in names}
    if len(lengths) != 1:
        raise ValueError(
            f"Probability vectors have mismatched lengths: {lengths}"
        )
    matrix = np.column_stack([np.asarray(prob_dict[n]) for n in names])
    return matrix, names


def train_meta_learner(
    train_probs: np.ndarray, y_train: ArrayLike
) -> LogisticRegression:
    """Fit a logistic-regression meta-learner on stacked base probabilities."""
    meta = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=ev.RANDOM_STATE,
    )
    meta.fit(train_probs, y_train)
    return meta


def evaluate_ensemble(
    meta: LogisticRegression, test_probs: np.ndarray, y_test: ArrayLike
) -> Dict[str, float]:
    """Evaluate the trained meta-learner on stacked test probabilities."""
    proba = meta.predict_proba(test_probs)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return ev.compute_metrics(y_test, preds, proba)


def meta_learner_weights(
    meta: LogisticRegression, names: Sequence[str]
) -> Dict[str, float]:
    """
    Return the meta-learner's learned coefficient per base model.

    A larger positive weight means the stack leans more heavily on that model
    when forming the final decision — useful for reporting *which* model the
    ensemble trusts most.
    """
    coefs = meta.coef_.ravel()
    if len(coefs) != len(names):
        raise ValueError(
            f"Coefficient count ({len(coefs)}) does not match model "
            f"count ({len(names)}): {list(names)}"
        )
    return {name: float(w) for name, w in zip(names, coefs)}


def soft_vote(
    prob_dict: ProbDict, weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Weighted average of base probabilities (no meta-learner).

    A robust, training-free fallback ensemble; handy as a comparison baseline.
    Pass ``weights`` as ``{model_name: weight}`` to favour stronger models;
    weights are normalised internally.
    """
    matrix, names = stack_probabilities(prob_dict)
    if weights is None:
        return matrix.mean(axis=1)
    w = np.asarray([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    return matrix @ w


def compare_strategies(
    train_prob_dict: ProbDict,
    y_train: ArrayLike,
    test_prob_dict: ProbDict,
    y_test: ArrayLike,
) -> Tuple[pd.DataFrame, LogisticRegression]:
    """
    Compare stacking vs soft-voting and report both.

    Guards against silent column misalignment: the train and test probability
    dicts must list the same models in the same order, otherwise the stacked
    matrices would line up the wrong columns and quietly corrupt results.

    Returns
    -------
    (results_df, fitted_meta_learner)
    """
    train_matrix, train_names = stack_probabilities(train_prob_dict)
    test_matrix, test_names = stack_probabilities(test_prob_dict)

    if train_names != test_names:
        raise ValueError(
            "Train/test model order mismatch — columns would misalign.\n"
            f"  train models: {train_names}\n"
            f"  test  models: {test_names}"
        )

    meta = train_meta_learner(train_matrix, y_train)
    stack_metrics = evaluate_ensemble(meta, test_matrix, y_test)

    vote_proba = soft_vote(test_prob_dict)
    vote_preds = (vote_proba >= 0.5).astype(int)
    vote_metrics = ev.compute_metrics(y_test, vote_preds, vote_proba)

    rows = [
        {"strategy": "Stacking (LR meta-learner)", **stack_metrics},
        {"strategy": "Soft voting (mean proba)", **vote_metrics},
    ]
    return pd.DataFrame(rows), meta


def load_probabilities(
    split: str,
    models: Sequence[str] = BASE_MODELS,
    prefix: str = "probs",
) -> Tuple[ProbDict, np.ndarray]:
    """
    Load and merge per-model probability exports for one split.

    Reads ``<prefix>_<model>_<split>.csv`` for each model — each file shaped
    ``uid, label, <model_name>`` — and merges them on ``uid`` so rows align by
    sample identity rather than by file order. This is the same contract
    ``eval_utils.save_probabilities`` writes from every model notebook.

    Parameters
    ----------
    split : str
        Typically ``"val"`` (stacking fold) or ``"test"`` (held-out fold).
    models : sequence of str
        Model names to load, in the order columns should appear.
    prefix : str
        Filename prefix; defaults to ``"probs"``.

    Returns
    -------
    (prob_dict, labels)
        ``prob_dict`` maps each model name to its positive-class probabilities;
        ``labels`` is the shared label vector (validated identical across files).
    """
    merged: Optional[pd.DataFrame] = None
    for model in models:
        path = f"{prefix}_{model}_{split}.csv"
        df = pd.read_csv(path)
        for col in ("uid", "label", model):
            if col not in df.columns:
                raise ValueError(
                    f"{path} is missing required column '{col}'. "
                    f"Found: {list(df.columns)}"
                )
        df = df[["uid", "label", model]]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["uid", "label"], how="inner")
        logger.info("Loaded %s (%d rows) from %s", model, len(df), path)

    if merged is None or merged.empty:
        raise ValueError(
            f"No rows after merging probability files for split '{split}'. "
            "Check that all model files share the same uids."
        )

    labels = merged["label"].to_numpy()
    prob_dict = {model: merged[model].to_numpy() for model in models}
    logger.info(
        "Merged %d samples across %d models for split '%s'.",
        len(merged), len(models), split,
    )
    return prob_dict, labels


def run_ensemble(
    models: Sequence[str] = BASE_MODELS,
    prefix: str = "probs",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    End-to-end ensemble run from the project's standard probability exports.

    Loads the ``val`` (stacking) and ``test`` (held-out) folds, trains the
    meta-learner on val, evaluates on test, and returns both the strategy
    comparison and the learned per-model weights. This is the one-call version
    of what the integrator notebook does step by step.

    Returns
    -------
    (results_df, weights)
        ``results_df`` compares stacking vs soft voting on the test fold;
        ``weights`` maps each model to its meta-learner coefficient.
    """
    train_prob_dict, y_train = load_probabilities("val", models, prefix)
    test_prob_dict, y_test = load_probabilities("test", models, prefix)

    results, meta = compare_strategies(
        train_prob_dict, y_train, test_prob_dict, y_test
    )
    weights = meta_learner_weights(meta, list(train_prob_dict))
    return results, weights


def main() -> None:
    """
    CLI entry point.

    Reads the standard ``probs_<model>_val.csv`` / ``probs_<model>_test.csv``
    exports from the current directory (or a chosen ``--prefix``), runs the
    full ensemble, and prints the strategy comparison plus the meta-learner's
    per-model weights.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="SafetyNet ensemble meta-learner")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(BASE_MODELS),
        help="Base model names to ensemble (default: bilstm svm_bert biobert).",
    )
    parser.add_argument(
        "--prefix",
        default="probs",
        help="Probability-file prefix (default: 'probs').",
    )
    args = parser.parse_args()

    results, weights = run_ensemble(models=args.models, prefix=args.prefix)

    print("\n=== Strategy comparison (test fold) ===")
    print(results.to_string(index=False))

    print("\n=== Meta-learner weights (higher = more trusted) ===")
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    for name, w in ranked:
        print(f"  {name:>10s}: {w:+.3f}")


if __name__ == "__main__":
    main()
