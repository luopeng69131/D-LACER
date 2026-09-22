"""Issue-time features used by the latent-state routers."""

from __future__ import annotations

import numpy as np


def _summarize_forecast(values: np.ndarray) -> np.ndarray:
    horizon = values.shape[1]
    axis = np.arange(horizon, dtype=np.float32)
    centered = axis - np.mean(axis)
    denominator = max(float(np.sum(centered**2)), 1.0)
    slope = np.sum(
        (values - np.mean(values, axis=1, keepdims=True)) * centered,
        axis=1,
    ) / denominator
    return np.column_stack(
        [
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.min(values, axis=1),
            np.max(values, axis=1),
            values[:, 0],
            values[:, -1],
            slope,
        ]
    )


def build_router_features(
    context: np.ndarray,
    candidate_predictions: np.ndarray,
) -> np.ndarray:
    """Build target-free context, forecast-shape, and disagreement features.

    Args:
        context: Issue-time context with shape ``(windows, features)``.
        candidate_predictions: Auxiliary forecasts with shape
            ``(experts, windows, horizon)``.
    """

    context = np.asarray(context, dtype=np.float32)
    candidate = np.asarray(candidate_predictions, dtype=np.float32)
    if context.ndim != 2:
        raise ValueError("context must have shape (windows, features)")
    if candidate.ndim != 3:
        raise ValueError(
            "candidate_predictions must have shape (experts, windows, horizon)"
        )
    if len(context) != candidate.shape[1]:
        raise ValueError("context and candidate predictions do not align")
    if candidate.shape[0] < 2:
        raise ValueError("at least two candidate forecasts are required")

    parts = [context]
    for expert_prediction in candidate:
        parts.append(_summarize_forecast(expert_prediction))

    anchor = candidate[0]
    for alternative in candidate[1:]:
        disagreement = alternative - anchor
        parts.append(_summarize_forecast(disagreement))
        parts.append(
            np.column_stack(
                [
                    np.mean(np.abs(disagreement), axis=1),
                    np.max(np.abs(disagreement), axis=1),
                    np.mean(alternative > anchor, axis=1),
                ]
            )
        )
    return np.column_stack(parts).astype(np.float32)
