"""Metrics over all items. An invalid prediction stays in the denominator and counts as wrong."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

from .labels import INVALID, LABELS, NAME_TO_ID

INVALID_ID = len(LABELS)  # column 6 of the confusion matrix


def to_ids(pred: Sequence[str]) -> np.ndarray:
    return np.array([NAME_TO_ID.get(p, INVALID_ID) for p in pred], dtype=int)


def bootstrap(stat, n: int, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, size=n)) for _ in range(n_boot)]
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def macro_f1(t: np.ndarray, p: np.ndarray) -> float:
    """Macro-F1 over the six labels from counts (an invalid prediction is a false negative)."""
    scores = []
    for c in range(len(LABELS)):
        tp = int(((t == c) & (p == c)).sum())
        fp = int(((t != c) & (p == c)).sum())
        fn = int(((t == c) & (p != c)).sum())
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[str], n_boot: int = 1000) -> dict:
    t = np.asarray(y_true, dtype=int)
    p = to_ids(y_pred)
    if t.shape != p.shape or t.size == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    labels = list(range(len(LABELS)))
    correct = t == p
    macro = lambda idx: macro_f1(t[idx], p[idx])  # noqa: E731
    prec, rec, f1, support = precision_recall_fscore_support(t, p, labels=labels, zero_division=0)
    cm = np.zeros((len(LABELS), len(LABELS) + 1), dtype=int)
    for a, b in zip(t, p):
        cm[a, b] += 1
    return {
        "n": int(t.size),
        "accuracy": float(correct.mean()),
        "accuracy_ci": bootstrap(lambda idx: correct[idx].mean(), t.size, n_boot),
        "macro_f1": float(macro(np.arange(t.size))),
        "macro_f1_ci": bootstrap(macro, t.size, n_boot),
        "weighted_f1": float(f1_score(t, p, labels=labels, average="weighted", zero_division=0)),
        "invalid_rate": float((p == INVALID_ID).mean()),
        "per_class": {LABELS[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                                  "support": int(support[i])} for i in labels},
        "confusion": {"rows": list(LABELS), "columns": [*LABELS, INVALID], "matrix": cm.tolist()},
    }


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Paired test of two systems on the same labelled items."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired results must have the same length")
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    return {"only_a_correct": only_a, "only_b_correct": only_b,
            "p_value": 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)}


def compare(y_true: Sequence[int], pred_a: Sequence[str], pred_b: Sequence[str], n_boot: int = 2000) -> dict:
    t = np.asarray(y_true, dtype=int)
    ca, cb = to_ids(pred_a) == t, to_ids(pred_b) == t
    diff = (ca.astype(float) - cb.astype(float))
    return {"accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
            "accuracy_diff": float(diff.mean()), "accuracy_diff_ci": bootstrap(lambda idx: diff[idx].mean(), t.size, n_boot),
            "mcnemar": mcnemar_exact(ca, cb)}


def agreement(pred_a: Sequence[str], pred_b: Sequence[str]) -> dict:
    """Agreement of two systems on unlabelled texts. No accuracy test is possible without labels."""
    a, b = to_ids(pred_a), to_ids(pred_b)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("predictions must have the same non-zero length")
    kappa = cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else math.nan
    return {"n": int(a.size), "agreement": float((a == b).mean()), "cohen_kappa": float(kappa),
            "invalid_a": float((a == INVALID_ID).mean()), "invalid_b": float((b == INVALID_ID).mean())}
