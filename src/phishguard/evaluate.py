"""Security-oriented metrics: performance at a fixed false-positive budget."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_for_fpr(y_true, y_score, max_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on (y_true, y_score) is <= max_fpr."""
    y_true = np.asarray(y_true)
    neg = np.sort(np.asarray(y_score)[y_true == 0])[::-1]
    k = int(np.floor(max_fpr * len(neg)))          # negatives we may flag
    if k >= len(neg):
        return 0.0
    return float(np.nextafter(neg[k], np.inf))       # strictly above the (k+1)-th highest negative


def tpr_at_fpr(y_true, y_score, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(max_fpr, fpr, tpr))


def metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    neg = y_true == 0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, y_score, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y_true, y_score, 0.001),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "fpr": float(y_pred[neg].mean()) if neg.any() else float("nan"),
        "n": int(len(y_true)),
    }
