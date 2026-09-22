import numpy as np
import pandas as pd

from dlacer.data import load_starnet_pickle


def test_window_construction(tmp_path) -> None:
    rows = 240
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="s"),
            "throughput": np.linspace(50, 100, rows),
            "latency": np.linspace(30, 40, rows),
            "alt": np.full(rows, 45.0),
            "az": np.linspace(0, 359, rows),
            "distance": np.full(rows, 800.0),
            "sat_name": [f"sat-{idx % 4}" for idx in range(rows)],
            "n_candidates": np.full(rows, 3),
            "clouds": np.zeros(rows),
            "pressure": np.full(rows, 1010.0),
            "humidity": np.full(rows, 60.0),
        }
    )
    source = tmp_path / "trace.pkl"
    frame.to_pickle(source)
    data = load_starnet_pickle(source, name="toy", step_len=5)

    assert data.X.shape[1] == 75 * 14
    assert data.y.shape[1] == 15
    assert len(data.X) == 31
    assert data.context.shape[0] == len(data.X)
    assert np.all(np.isfinite(data.X))
