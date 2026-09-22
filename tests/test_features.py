import numpy as np

from dlacer.features import build_router_features


def test_router_features_are_target_free_and_aligned() -> None:
    context = np.arange(24, dtype=np.float32).reshape(8, 3)
    candidates = np.stack(
        [
            np.ones((8, 5), dtype=np.float32),
            np.full((8, 5), 2.0, dtype=np.float32),
        ]
    )
    features = build_router_features(context, candidates)

    assert features.shape[0] == 8
    assert features.shape[1] > context.shape[1]
    assert np.all(np.isfinite(features))
