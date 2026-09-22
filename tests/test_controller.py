import numpy as np

from dlacer.controller import DLACERConfig, DLACERController
from dlacer.groups import RiskGroups


def test_controller_returns_group_metrics_and_finite_predictions() -> None:
    n_windows, horizon = 60, 6
    target = np.full((n_windows, horizon), 40.0, dtype=np.float32)
    base = target + 5.0
    global_mask = np.ones_like(target, dtype=bool)
    low_mask = np.broadcast_to(
        (np.arange(n_windows) % 2 == 0)[:, None], target.shape
    )
    near_mask = np.zeros_like(target, dtype=bool)
    near_mask[:, : horizon // 2] = True
    groups = RiskGroups(
        names=("global", "latent_low_capacity", "near_horizon"),
        mask=np.stack([global_mask, low_mask, near_mask], axis=2),
        thresholds={},
    )

    result = DLACERController(
        DLACERConfig(learning_rate=1.0, target_margin=0.0)
    ).run(
        base,
        target,
        groups,
        budget=0.35,
        initialization_end=20,
        evaluation_start=35,
        feedback_delay=3,
    )

    assert result.prediction.shape == target.shape
    assert np.all(np.isfinite(result.prediction[20:]))
    assert set(result.group_metrics) == set(groups.names)
    assert result.metrics["overestimation_rate"] < 1.0
    assert set(result.group_passes(0.35)) == set(groups.names)
