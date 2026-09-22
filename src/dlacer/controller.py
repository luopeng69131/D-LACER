"""Delayed projected control for overlapping conditional risk groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .groups import RiskGroups
from .metrics import forecasting_metrics


@dataclass(frozen=True)
class DLACERConfig:
    """Hyperparameters of the D-LACER safety layer."""

    learning_rate: float = 2.0
    target_margin: float = 0.02
    correction_min: float = -200.0
    correction_max: float = 300.0


@dataclass
class DLACERResult:
    """Predictions, controller states, and aggregate/group metrics."""

    prediction: np.ndarray
    correction_trace: np.ndarray
    metrics: dict[str, float]
    group_metrics: dict[str, dict[str, float]]

    def group_passes(self, budget: float) -> dict[str, bool]:
        """Report empirical budget compliance for each conditional group."""

        return {
            name: values["overestimation_rate"] <= budget
            for name, values in self.group_metrics.items()
        }


def _boundary_correction(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    budget: float,
) -> float:
    residual = (prediction - target)[mask]
    if len(residual) == 0:
        return 0.0
    allowed = int(np.floor(budget * len(residual)))
    rank = max(len(residual) - allowed - 1, 0)
    return float(np.sort(residual)[rank])


class DLACERController:
    """Apply group-specific risk corrections with delayed observations."""

    def __init__(self, config: DLACERConfig | None = None):
        self.config = config or DLACERConfig()

    def run(
        self,
        base_prediction: np.ndarray,
        target: np.ndarray,
        groups: RiskGroups,
        budget: float,
        initialization_end: int,
        evaluation_start: int,
        feedback_delay: int,
    ) -> DLACERResult:
        """Run D-LACER chronologically over a forecast stream.

        ``target`` is read by the online update only after ``feedback_delay``
        windows. It is also used before ``initialization_end`` to initialize
        each group correction from historical residuals.
        """

        base = np.asarray(base_prediction, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if base.shape != target.shape or groups.mask.shape[:2] != target.shape:
            raise ValueError("predictions, targets, and group masks do not align")
        if len(groups.names) != groups.mask.shape[2]:
            raise ValueError("group names and group mask do not align")
        if not 0 < budget < 1:
            raise ValueError("budget must lie between zero and one")
        if not 0 < initialization_end <= evaluation_start < len(target):
            raise ValueError("invalid chronological boundaries")
        if feedback_delay < 1:
            raise ValueError("feedback_delay must be positive")

        n_windows = len(target)
        n_groups = len(groups.names)
        target_budget = max(float(budget) - self.config.target_margin, 0.01)
        corrections = np.asarray(
            [
                _boundary_correction(
                    base[:initialization_end],
                    target[:initialization_end],
                    groups.mask[:initialization_end, :, group_idx],
                    target_budget,
                )
                for group_idx in range(n_groups)
            ],
            dtype=np.float64,
        )

        prediction = np.full_like(target, np.nan, dtype=np.float32)
        correction_trace = np.full((n_windows, n_groups), np.nan, dtype=np.float32)
        for window_idx in range(initialization_end, n_windows):
            feedback_idx = window_idx - feedback_delay
            if feedback_idx >= initialization_end:
                for group_idx in range(n_groups):
                    membership = groups.mask[feedback_idx, :, group_idx]
                    if not np.any(membership):
                        continue
                    realized_risk = float(
                        np.mean(
                            prediction[feedback_idx, membership]
                            > target[feedback_idx, membership]
                        )
                    )
                    corrections[group_idx] = np.clip(
                        corrections[group_idx]
                        + self.config.learning_rate
                        * (realized_risk - target_budget),
                        self.config.correction_min,
                        self.config.correction_max,
                    )

            active = groups.mask[window_idx]
            element_correction = np.max(
                np.where(active, corrections[None, :], -np.inf),
                axis=1,
            )
            if not np.all(np.isfinite(element_correction)):
                raise ValueError("every forecast element must belong to an active group")
            prediction[window_idx] = base[window_idx] - element_correction
            correction_trace[window_idx] = corrections

        evaluation = np.arange(evaluation_start, n_windows)
        group_metrics: dict[str, dict[str, float]] = {}
        for group_idx, name in enumerate(groups.names):
            membership = groups.mask[evaluation, :, group_idx]
            if np.any(membership):
                group_metrics[name] = forecasting_metrics(
                    prediction[evaluation][membership],
                    target[evaluation][membership],
                )
        return DLACERResult(
            prediction=prediction,
            correction_trace=correction_trace,
            metrics=forecasting_metrics(prediction[evaluation], target[evaluation]),
            group_metrics=group_metrics,
        )
