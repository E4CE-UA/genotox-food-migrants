"""Baselines, metrics suited to imbalance and calibration.

With a prevalence of positives around 5%, accuracy says nothing:
always predicting 'not genotoxic' is correct 95% of the time. That is why the
summary uses PR-AUC (whose baseline is prevalence, not 0.5), MCC and positive
recall, and that is why the majority class baseline always appears in the
table: if a model does not beat it on PR-AUC, it has not learned anything useful.

MCC is given at two thresholds: 0.5, and the threshold that makes the number of
predicted positives equal to the actual number of positives in the test (prevalence
threshold). The first is usually degenerate with imbalanced classes; the second
measures the ranking without giving the model the choice of threshold.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

MODELS = ("majority_class", "logistic", "lgbm")


def build(name: str, seed: int):
    if name == "majority_class":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return LogisticRegression(
            max_iter=5000, C=1.0, class_weight="balanced", solver="liblinear",
            random_state=seed,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    raise ValueError(f"unknown model: {name!r} (options: {MODELS})")


def prevalence_threshold(p, n_pos: int) -> float:
    """Threshold that leaves n_pos elements above it (informative, may tie)."""
    if n_pos <= 0 or n_pos >= len(p):
        return 0.5
    return float(np.sort(p)[::-1][n_pos - 1])


def top_n_prediction(p, n_pos: int) -> np.ndarray:
    """Marks as positive exactly the n_pos with the highest probability.

    Selection is by rank rather than by comparing to the threshold, because a
    constant predictor (the majority class baseline) ties on all values and
    with a >= threshold comparison would end up predicting everything positive,
    which gives a misleading recall of 1.0.
    """
    p = np.asarray(p, dtype=float)
    pred = np.zeros(len(p), dtype=int)
    if n_pos > 0:
        pred[np.argsort(-p, kind="stable")[:min(n_pos, len(p))]] = 1
    return pred


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    prev = y.mean() if len(y) else np.nan
    pred_prev = top_n_prediction(p, n_pos)
    unique = len(np.unique(y)) < 2
    return {
        "n_test": len(y),
        "pos_test": n_pos,
        "prevalence": round(float(prev), 4),
        "pr_auc": np.nan if unique else round(float(average_precision_score(y, p)), 4),
        "pr_auc_lift": np.nan if unique or prev == 0
        else round(float(average_precision_score(y, p)) / float(prev), 2),
        "roc_auc": np.nan if unique else round(float(roc_auc_score(y, p)), 4),
        "mcc_050": round(float(matthews_corrcoef(y, (p >= 0.5).astype(int))), 4),
        "mcc_prev": round(float(matthews_corrcoef(y, pred_prev)), 4),
        "recall_pos_050": round(float(recall_score(y, (p >= 0.5).astype(int), zero_division=0)), 4),
        "recall_pos_prev": round(float(recall_score(y, pred_prev, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(ece(y, p), 4),
    }


def evaluate(X, y, splits, model_names=MODELS) -> tuple:
    """Trains and evaluates each model on each split.

    Returns (table of metrics per model and seed, dict of predictions
    {(model, seed): (y_test, p_test, idx_test)}) so the probabilities can be
    reused in calibration and in the applicability domain.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    rows, predictions = [], {}
    for seed, tr, te in splits:
        for name in model_names:
            clf = build(name, seed).fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            predictions[(name, seed)] = (y[te], p, te)
            rows.append({"model": name, "seed": seed, **metrics(y[te], p)})
    return pd.DataFrame(rows), predictions


def median_iqr(res: pd.DataFrame, columns=("pr_auc", "roc_auc", "mcc_prev",
                                             "recall_pos_prev", "ece")) -> pd.DataFrame:
    """Median and IQR over the seeds: one row per model."""
    rows = []
    for model, g in res.groupby("model", sort=False):
        row = {"model": model, "n_seeds": len(g)}
        for c in columns:
            q1, med, q3 = g[c].quantile([0.25, 0.5, 0.75])
            row[c] = f"{med:.3f} [{q1:.3f}-{q3:.3f}]"
        rows.append(row)
    order = {m: i for i, m in enumerate(MODELS)}
    return pd.DataFrame(rows).sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


# ------------------------------------------------------------- calibration
def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def domain_curve(similarities, y, p, thresholds=None, n_minimum: int = 20) -> pd.DataFrame:
    """PR-AUC of the subset that remains within domain, threshold by threshold."""
    similarities = np.asarray(similarities, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    thresholds = np.arange(0.1, 0.85, 0.05) if thresholds is None else np.asarray(thresholds)
    rows = []
    for u in thresholds:
        m = similarities >= u
        evaluable = m.sum() >= n_minimum and len(np.unique(y[m])) > 1
        rows.append({
            "threshold": round(float(u), 2),
            "n": int(m.sum()),
            "coverage": float(m.mean()),
            "pr_auc": round(float(average_precision_score(y[m], p[m])), 4) if evaluable else np.nan,
            "prevalence": round(float(y[m].mean()), 4) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def reliability_curve(y, p, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(m.sum()),
            "p_mean": float(p[m].mean()) if m.any() else np.nan,
            "frac_positives": float(y[m].mean()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def call_threshold_curve(p, y, thresholds=None) -> pd.DataFrame:
    """What a calling threshold would call, and what those calls are worth.

    Computed on held-out scores. The grid is quantiles of the scores rather
    than an even spacing on [0, 1], because the scores are compressed hard
    toward zero and an even grid would spend forty points on an empty range.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if thresholds is None:
        thresholds = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 40)))
    n_pos = max(int(y.sum()), 1)
    rows = []
    for t in thresholds:
        called = p >= t
        n_called = int(called.sum())
        true_pos = int((called & (y == 1)).sum())
        rows.append({
            "threshold": float(t),
            "n_called": n_called,
            "call_rate": n_called / len(p),
            "precision": true_pos / n_called if n_called else float("nan"),
            "recall": true_pos / n_pos,
        })
    return pd.DataFrame(rows)
