"""Binary evaluation with borderline sensitivity scenarios."""

from __future__ import annotations

import numpy as np


def roc_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FPR/TPR at each distinct score, including the no-positive point."""
    order = np.argsort(-p, kind="stable")
    sorted_scores = p[order]
    sorted_labels = y[order]
    last = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tp = np.cumsum(sorted_labels)[last]
    fp = 1 + last - tp
    return np.r_[0.0, fp / np.sum(y == 0)], np.r_[0.0, tp / np.sum(y == 1)]


def binary_metrics(labels, probabilities, *, threshold: float = 0.5, target_specificity: float = 0.95) -> dict:
    y = np.asarray(labels, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p) or not len(y):
        raise ValueError("Nonempty labels and probabilities must have equal length")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation needs both binary classes")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not 0 < target_specificity < 1:
        raise ValueError("target_specificity must be in (0, 1)")
    predicted = (p >= threshold).astype(int)
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    fpr, tpr = roc_points(y, p)
    eligible = np.flatnonzero(fpr <= 1 - target_specificity + 1e-12)
    return {
        "n_images": int(len(y)),
        "auroc": float(np.trapezoid(tpr, fpr)),
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "target_specificity": float(target_specificity),
        "sensitivity_at_target_specificity": float(np.max(tpr[eligible])),
    }


def borderline_scenarios(labels, probabilities, **kwargs) -> dict[str, dict]:
    """Treat label 2 as excluded, healthy, or glaucoma; never fit on test data."""
    y = np.asarray(labels, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p) or not set(np.unique(y)).issubset({0, 1, 2}):
        raise ValueError("Expected aligned labels 0, 1, or 2")
    scenarios = {
        "borderline_excluded": (y != 2, y),
        "borderline_as_healthy": (np.ones(len(y), dtype=bool), np.where(y == 2, 0, y)),
        "borderline_as_glaucoma": (np.ones(len(y), dtype=bool), np.where(y == 2, 1, y)),
    }
    return {name: binary_metrics(mapped[mask], p[mask], **kwargs)
            for name, (mask, mapped) in scenarios.items()}
