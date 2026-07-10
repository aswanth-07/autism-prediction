"""Evaluation helpers with threshold selection isolated to training predictions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from asd_screening.config import RANDOM_STATE


def positive_probabilities(estimator: Any, features: pd.DataFrame) -> NDArray[np.float64]:
    """Extract positive-class probabilities from a fitted classifier pipeline."""

    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Selected estimator does not expose predict_proba")
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected binary class probabilities with shape (n_samples, 2)")
    return probabilities[:, 1]


def select_f1_threshold(target: pd.Series, scores: NDArray[np.float64]) -> tuple[float, float]:
    """Choose an F1-maximizing threshold using out-of-fold training scores only."""

    thresholds = np.linspace(0.10, 0.90, 161)
    f1_values = np.array(
        [f1_score(target, scores >= threshold, zero_division=0) for threshold in thresholds]
    )
    best = float(np.max(f1_values))
    tied = thresholds[np.isclose(f1_values, best)]
    threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    return threshold, best


def classification_metrics(
    target: pd.Series | NDArray[np.integer[Any]],
    scores: NDArray[np.float64],
    threshold: float,
) -> dict[str, Any]:
    """Calculate complementary discrimination and classification metrics."""

    truth = np.asarray(target, dtype=int)
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "brier_score": float(brier_score_loss(truth, scores)),
        "confusion_matrix": matrix.tolist(),
    }


def bootstrap_interval(
    target: pd.Series | NDArray[np.integer[Any]],
    scores: NDArray[np.float64] | NDArray[np.integer[Any]],
    metric: Callable[
        [NDArray[np.integer[Any]], NDArray[np.float64] | NDArray[np.integer[Any]]], float
    ],
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a stratified-free percentile bootstrap interval for a holdout metric."""

    truth = np.asarray(target, dtype=int)
    rng = np.random.default_rng(RANDOM_STATE)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(truth), len(truth))
        sampled_truth = truth[indices]
        if np.unique(sampled_truth).size < 2:
            continue
        values.append(float(metric(sampled_truth, scores[indices])))
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)
