#!/usr/bin/env python3
"""Run D-LACER on a small synthetic delayed-feedback stream."""

from __future__ import annotations

import json

import numpy as np

from dlacer import DLACERController, RouterConfig, build_latent_risk_groups


def make_stream(seed: int = 7) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    n_windows, horizon = 260, 15
    time = np.arange(n_windows, dtype=np.float32)
    capacity = 90.0 + 28.0 * np.sin(time / 18.0) + 10.0 * np.sin(time / 5.5)
    context = np.column_stack(
        [
            capacity + rng.normal(0, 4, n_windows),
            np.sin(time / 18.0),
            np.cos(time / 18.0),
            rng.normal(size=n_windows),
        ]
    ).astype(np.float32)
    horizon_decay = np.linspace(0.0, 6.0, horizon, dtype=np.float32)
    target = capacity[:, None] - horizon_decay + rng.normal(0, 7, (n_windows, horizon))
    point = capacity[:, None] - horizon_decay + 4.0 + rng.normal(
        0, 5, (n_windows, horizon)
    )
    conservative = point - 10.0 + rng.normal(0, 1.5, point.shape)
    adaptive = point - 5.0 - 0.12 * np.maximum(95.0 - capacity[:, None], 0.0)
    candidates = np.stack([point, conservative, adaptive]).astype(np.float32)
    return context, point.astype(np.float32), candidates, target.astype(np.float32)


def main() -> None:
    context, point, candidates, target = make_stream()
    initialization_end = 104
    evaluation_start = 156
    budget = 0.35
    groups = build_latent_risk_groups(
        context,
        point,
        candidates,
        target,
        initialization_end,
        RouterConfig(n_estimators=50, n_jobs=1),
    )
    result = DLACERController().run(
        point,
        target,
        groups,
        budget=budget,
        initialization_end=initialization_end,
        evaluation_start=evaluation_start,
        feedback_delay=15,
    )
    summary = {
        "budget": budget,
        "metrics": result.metrics,
        "group_overestimation_rate": {
            name: values["overestimation_rate"]
            for name, values in result.group_metrics.items()
        },
        "group_budget_pass": result.group_passes(budget),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
