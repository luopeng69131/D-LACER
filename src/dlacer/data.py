"""Starlink trace cleaning, window construction, and temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_COLUMNS = (
    "throughput",
    "latency",
    "alt",
    "az",
    "distance",
    "sat_name",
    "n_candidates",
    "clouds",
    "pressure",
    "humidity",
    "t_s",
    "t_minute",
    "t_hour",
    "t_d_of_w",
)


@dataclass
class WindowedData:
    X: np.ndarray
    y: np.ndarray
    context: np.ndarray
    sample_times: np.ndarray
    starts: np.ndarray
    feature_names: list[str]
    context_feature_names: list[str]
    input_len: int
    output_len: int
    step_len: int
    name: str = "dataset"


@dataclass(frozen=True)
class DataSplits:
    base_train: np.ndarray
    controller: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    purge_windows: int

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "base_train": self.base_train,
            "controller": self.controller,
            "calibration": self.calibration,
            "test": self.test,
        }


def _linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(x_centered * (values - values.mean())) / denominator)


def _context_features(
    throughput_window: np.ndarray,
    last_row: np.ndarray,
    exogenous_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    features: list[float] = []
    names: list[str] = []
    window_length = len(throughput_window)
    for requested_tail in (5, 15, window_length):
        tail_size = min(requested_tail, window_length)
        tail = throughput_window[-tail_size:]
        prefix = "tp_all" if requested_tail == window_length else f"tp_{requested_tail}"
        values = (
            float(tail[-1]),
            float(np.mean(tail)),
            float(np.std(tail)),
            float(np.min(tail)),
            float(np.max(tail)),
            float(np.quantile(tail, 0.10)),
            float(np.quantile(tail, 0.90)),
            _linear_slope(tail),
        )
        suffixes = ("last", "mean", "std", "min", "max", "p10", "p90", "slope")
        features.extend(values)
        names.extend(f"{prefix}_{suffix}" for suffix in suffixes)
    features.extend(
        [
            float(throughput_window[-1] - throughput_window[0]),
            float(np.max(throughput_window) - np.min(throughput_window)),
        ]
    )
    names.extend(("tp_window_delta", "tp_window_range"))
    features.extend(float(value) for value in last_row)
    names.extend(f"last_{name}" for name in exogenous_names)
    return np.asarray(features, dtype=np.float32), names


def load_starnet_pickle(
    path: str | Path,
    *,
    name: str,
    step_len: int,
    input_len: int = 75,
    output_len: int = 15,
) -> WindowedData:
    """Read a Starlink trace pickle and construct paper-aligned windows."""

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    if min(step_len, input_len, output_len) < 1:
        raise ValueError("step_len, input_len, and output_len must be positive")

    frame = pd.read_pickle(path).copy()
    required = {"timestamp", *MODEL_COLUMNS[:10]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["t_s"] = (frame["timestamp"].dt.second - 12) % 15
    frame["t_d_of_w"] = frame["timestamp"].dt.dayofweek
    frame["t_minute"] = frame["timestamp"].dt.minute
    frame["t_hour"] = frame["timestamp"].dt.hour
    frame["sat_name"] = pd.factorize(frame["sat_name"], sort=True)[0]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["timestamp", *MODEL_COLUMNS]).reset_index(drop=True)

    values = frame.loc[:, MODEL_COLUMNS].to_numpy(dtype=np.float32)
    throughput = frame["throughput"].to_numpy(dtype=np.float32)
    timestamps = frame["timestamp"].astype(str).to_numpy(dtype=str)
    max_start = len(frame) - input_len - output_len
    if max_start < 0:
        raise ValueError(
            f"{name} has {len(frame)} rows; at least {input_len + output_len} are needed"
        )
    starts = np.arange(0, max_start + 1, step_len, dtype=np.int64)

    n_samples = len(starts)
    n_columns = len(MODEL_COLUMNS)
    X = np.empty((n_samples, input_len * n_columns), dtype=np.float32)
    y = np.empty((n_samples, output_len), dtype=np.float32)
    sample_times = np.empty(n_samples, dtype=f"<U{max(len(value) for value in timestamps)}")
    exogenous_names = list(MODEL_COLUMNS[1:])
    first_context, context_names = _context_features(
        throughput[:input_len], values[input_len - 1, 1:], exogenous_names
    )
    context = np.empty((n_samples, len(first_context)), dtype=np.float32)

    for row_idx, start in enumerate(starts):
        input_end = start + input_len
        output_end = input_end + output_len
        input_values = values[start:input_end]
        X[row_idx] = input_values.reshape(-1)
        y[row_idx] = throughput[input_end:output_end]
        context[row_idx], _ = _context_features(
            throughput[start:input_end], input_values[-1, 1:], exogenous_names
        )
        sample_times[row_idx] = timestamps[input_end]

    feature_names = [
        f"{column}_lag_{input_len - step}"
        for step in range(input_len)
        for column in MODEL_COLUMNS
    ]
    return WindowedData(
        X=X,
        y=y,
        context=context,
        sample_times=sample_times,
        starts=starts,
        feature_names=feature_names,
        context_feature_names=context_names,
        input_len=input_len,
        output_len=output_len,
        step_len=step_len,
        name=name,
    )


def chronological_splits(
    data: WindowedData,
    fractions: tuple[float, float, float, float] = (0.55, 0.15, 0.15, 0.15),
) -> DataSplits:
    """Create four purged chronological splits without overlapping windows."""

    fractions_array = np.asarray(fractions, dtype=np.float64)
    if len(fractions_array) != 4 or np.any(fractions_array <= 0):
        raise ValueError("fractions must contain four positive values")
    fractions_array /= fractions_array.sum()
    purge_windows = max(
        1,
        int(np.ceil((data.input_len + data.output_len) / data.step_len)),
    )
    usable = len(data.X) - 3 * purge_windows
    if usable < 40:
        raise ValueError("not enough windows for four purged chronological splits")
    counts = np.floor(usable * fractions_array).astype(int)
    counts[-1] += usable - int(counts.sum())

    splits: list[np.ndarray] = []
    cursor = 0
    for split_idx, count in enumerate(counts):
        splits.append(np.arange(cursor, cursor + count, dtype=np.int64))
        cursor += count
        if split_idx < 3:
            cursor += purge_windows
    return DataSplits(
        base_train=splits[0],
        controller=splits[1],
        calibration=splits[2],
        test=splits[3],
        purge_windows=purge_windows,
    )


def save_windowed_data(data: WindowedData, path: str | Path) -> Path:
    """Save prepared arrays in a portable compressed NumPy archive."""

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=data.X,
        y=data.y,
        context=data.context,
        sample_times=data.sample_times,
        starts=data.starts,
        feature_names=np.asarray(data.feature_names, dtype=str),
        context_feature_names=np.asarray(data.context_feature_names, dtype=str),
        input_len=np.asarray(data.input_len),
        output_len=np.asarray(data.output_len),
        step_len=np.asarray(data.step_len),
        name=np.asarray(data.name),
    )
    return path


def load_windowed_archive(path: str | Path) -> WindowedData:
    """Load an archive produced by :func:`save_windowed_data`."""

    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        return WindowedData(
            X=archive["X"].astype(np.float32),
            y=archive["y"].astype(np.float32),
            context=archive["context"].astype(np.float32),
            sample_times=archive["sample_times"].astype(str),
            starts=archive["starts"].astype(np.int64),
            feature_names=archive["feature_names"].astype(str).tolist(),
            context_feature_names=archive["context_feature_names"].astype(str).tolist(),
            input_len=int(archive["input_len"]),
            output_len=int(archive["output_len"]),
            step_len=int(archive["step_len"]),
            name=str(archive["name"]),
        )
