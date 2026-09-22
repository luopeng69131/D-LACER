"""Issue-time construction of overlapping conditional risk groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xgboost import XGBClassifier

from .features import build_router_features


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for the two latent capacity-state routers."""

    low_capacity_quantile: float = 0.30
    severe_capacity_quantile: float = 0.10
    disagreement_quantile: float = 0.70
    fit_fraction: float = 0.75
    n_estimators: int = 240
    max_depth: int = 4
    learning_rate: float = 0.04
    min_child_weight: float = 10.0
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 8.0
    n_jobs: int = -1
    random_state: int = 42


@dataclass(frozen=True)
class RiskGroups:
    """Overlapping memberships with shape ``(windows, horizon, groups)``."""

    names: tuple[str, ...]
    mask: np.ndarray
    thresholds: dict[str, float]

    def subset(self, stop: int) -> "RiskGroups":
        """Return the leading windows while preserving group metadata."""

        return RiskGroups(self.names, self.mask[:stop], dict(self.thresholds))


def _validate_inputs(
    context: np.ndarray,
    point_prediction: np.ndarray,
    candidate_predictions: np.ndarray,
    target: np.ndarray,
    initialization_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    context = np.asarray(context, dtype=np.float32)
    point = np.asarray(point_prediction, dtype=np.float32)
    candidate = np.asarray(candidate_predictions, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if point.ndim != 2 or target.shape != point.shape:
        raise ValueError("point_prediction and target must share (windows, horizon)")
    if candidate.ndim != 3 or candidate.shape[1:] != point.shape:
        raise ValueError("candidate predictions do not align with point predictions")
    if context.ndim != 2 or len(context) != len(point):
        raise ValueError("context and forecast windows do not align")
    if not 1 <= initialization_end < len(point):
        raise ValueError("initialization_end must lie inside the sequence")
    return context, point, candidate, target


def build_latent_risk_groups(
    context: np.ndarray,
    point_prediction: np.ndarray,
    candidate_predictions: np.ndarray,
    target: np.ndarray,
    initialization_end: int,
    config: RouterConfig | None = None,
) -> RiskGroups:
    """Identify seven overlapping risk groups using issue-time information.

    Historical targets before ``initialization_end`` train two routers for
    latent low-capacity states. Membership after deployment is determined only
    from issue-time context and auxiliary forecasts.
    """

    config = config or RouterConfig()
    context, point, candidate, target = _validate_inputs(
        context,
        point_prediction,
        candidate_predictions,
        target,
        initialization_end,
    )
    n_windows, horizon = point.shape
    router_fit_end = int(config.fit_fraction * initialization_end)
    if router_fit_end < 20 or router_fit_end >= initialization_end:
        raise ValueError("initialization segment is too short for latent routing")

    features = build_router_features(context, candidate)
    target_level = np.mean(target, axis=1)
    masks: list[np.ndarray] = [np.ones((n_windows, horizon), dtype=bool)]
    names = ["global"]
    thresholds: dict[str, float] = {}
    parameters = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "tree_method": "hist",
        "n_jobs": config.n_jobs,
        "random_state": config.random_state,
        "verbosity": 0,
        "eval_metric": "logloss",
    }
    capacity_groups = (
        (config.low_capacity_quantile, "latent_low_capacity"),
        (config.severe_capacity_quantile, "latent_severe_capacity"),
    )
    for quantile, name in capacity_groups:
        capacity_threshold = float(np.quantile(target_level[:router_fit_end], quantile))
        historical_label = (target_level[:router_fit_end] <= capacity_threshold).astype(
            np.int8
        )
        if len(np.unique(historical_label)) < 2:
            raise ValueError(f"historical data cannot identify both classes for {name}")
        router = XGBClassifier(**parameters)
        router.fit(features[:router_fit_end], historical_label)
        probability = router.predict_proba(features)[:, 1]
        probability_threshold = float(
            np.quantile(
                probability[router_fit_end:initialization_end],
                1.0 - quantile,
            )
        )
        membership = probability >= probability_threshold
        masks.append(np.broadcast_to(membership[:, None], (n_windows, horizon)))
        names.append(name)
        thresholds[f"{name}_probability"] = probability_threshold
        thresholds[f"{name}_historical_capacity"] = capacity_threshold

    disagreement = np.mean(np.std(candidate, axis=0), axis=1)
    disagreement_threshold = float(
        np.quantile(disagreement[:initialization_end], config.disagreement_quantile)
    )
    masks.append(
        np.broadcast_to(
            (disagreement >= disagreement_threshold)[:, None],
            (n_windows, horizon),
        )
    )
    names.append("high_expert_disagreement")
    thresholds["high_expert_disagreement"] = disagreement_threshold

    edges = np.linspace(0, horizon, 4, dtype=int)
    for group_idx, name in enumerate(("near_horizon", "mid_horizon", "far_horizon")):
        mask = np.zeros((n_windows, horizon), dtype=bool)
        mask[:, edges[group_idx] : edges[group_idx + 1]] = True
        names.append(name)
        masks.append(mask)

    return RiskGroups(
        names=tuple(names),
        mask=np.stack(masks, axis=2),
        thresholds=thresholds,
    )
