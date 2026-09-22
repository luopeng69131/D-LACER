"""Forecast accuracy and one-sided safety metrics."""

from __future__ import annotations

import numpy as np


def forecasting_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute symmetric errors and positive-error safety metrics."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if prediction.size == 0:
        raise ValueError("prediction and target must be non-empty")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("prediction and target must contain only finite values")

    error = prediction - target
    positive_error = np.maximum(error, 0.0)
    negative_error = np.maximum(-error, 0.0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "overestimation_rate": float(np.mean(error > 0.0)),
        "mean_positive_error": float(np.mean(positive_error)),
        "p95_positive_error": float(np.percentile(positive_error, 95)),
        "mean_negative_error": float(np.mean(negative_error)),
    }


def budget_feasible(
    metrics: dict[str, float],
    budget: float,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether a metric record satisfies an overestimation budget."""

    return metrics["overestimation_rate"] <= budget + tolerance
