"""Cross-domain experiment harness for SafetyNet AI.

Runs the full train-domain x test-domain matrix (e.g. Formal->Formal,
Formal->Informal, Informal->Formal, Informal->Informal) for any model
exposed as a fit_predict callable, then summarises the in-domain vs
cross-domain degradation. Model-agnostic: works for Bi-LSTM, SVM-BERT,
BioBERT, or the ensemble.
"""
import numpy as np
import pandas as pd
import eval_utils as ev


def _match(series, name):
    """Match a domain by exact value, else case-insensitive substring.

    Robust to values like 'Formal (DailyMed)' when asked for 'Formal'.
    """
    s = series.astype(str)
    exact = s == name
    if exact.any():
        return exact
    return s.str.contains(name, case=False, na=False)


def run_domain_matrix(full_df, fit_predict_fn, domains=("Formal", "Informal"),
                      domain_col="domain"):
    """Run every (train_domain, test_domain) combination.

    Args:
        full_df: DataFrame with a domain column and the model's inputs.
        fit_predict_fn: callable(train_df, test_df) -> dict with keys
            'y_true', 'y_pred', and optionally 'y_proba'.
        domains: iterable of domain names to cross.
        domain_col: name of the domain column.

    Returns:
        (results_df, preds): results_df has one row per combination with
        train_domain, test_domain, in_domain flag, and all metrics; preds
        maps (train, test) -> dict of arrays for confusion matrices.
    """
    rows, preds = [], {}
    for tr in domains:
        train_df = full_df[_match(full_df[domain_col], tr)]
        for te in domains:
            test_df = full_df[_match(full_df[domain_col], te)]
            out = fit_predict_fn(train_df, test_df)
            y_true = np.asarray(out["y_true"])
            y_pred = np.asarray(out["y_pred"])
            proba = out.get("y_proba")
            metrics = ev.compute_metrics(y_true, y_pred, proba)
            rows.append({"train_domain": tr, "test_domain": te,
                         "in_domain": tr == te, **metrics})
            preds[(tr, te)] = {"y_true": y_true, "y_pred": y_pred,
                               "y_proba": proba}
    return pd.DataFrame(rows), preds


def matrix_grid(results_df, metric="f1"):
    """Pivot results into a train x test grid for a heatmap."""
    return results_df.pivot(index="train_domain", columns="test_domain",
                            values=metric)


def degradation_summary(results_df, metric="f1"):
    """Per train-domain in-domain vs cross-domain score and the drop."""
    out = []
    for tr, grp in results_df.groupby("train_domain"):
        in_d = grp[grp["in_domain"]][metric].mean()
        cross = grp[~grp["in_domain"]][metric].mean()
        out.append({
            "train_domain": tr,
            f"in_domain_{metric}": round(in_d, 4),
            f"cross_domain_{metric}": round(cross, 4),
            "drop": round(in_d - cross, 4),
            "drop_pct": round((in_d - cross) / in_d * 100, 2)
            if in_d else float("nan"),
        })
    return pd.DataFrame(out)


def combine_model_matrices(named_results, metric="f1"):
    """Stack several models' results into one model x combo table.

    Args:
        named_results: dict of model_name -> results_df.
        metric: which metric to report.

    Returns:
        DataFrame indexed by model, columns 'train->test' combinations.
    """
    rows = {}
    for name, res in named_results.items():
        rows[name] = {
            f"{r.train_domain[:1]}->{r.test_domain[:1]}": getattr(r, metric)
            for r in res.itertuples()
        }
    return pd.DataFrame(rows).T
