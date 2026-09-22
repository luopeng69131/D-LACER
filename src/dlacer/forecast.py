"""A compact XGBoost forecast bank for the end-to-end example."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ForecastBankConfig:
    quantiles: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    n_estimators: int = 120
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    device: str = "cpu"


class XGBoostForecastBank:
    """Train one point and one multi-quantile model per horizon."""

    def __init__(self, output_len: int, config: ForecastBankConfig | None = None):
        self.output_len = int(output_len)
        self.config = config or ForecastBankConfig()
        self.quantiles = np.asarray(self.config.quantiles, dtype=np.float32)
        if len(self.quantiles) < 2 or np.any(np.diff(self.quantiles) <= 0):
            raise ValueError("quantiles must be a strictly increasing sequence")
        self.point_models: list[XGBRegressor] = []
        self.quantile_models: list[XGBRegressor] = []

    def _common_params(self) -> dict[str, object]:
        return {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_lambda": self.config.reg_lambda,
            "min_child_weight": self.config.min_child_weight,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            "tree_method": "hist",
            "device": self.config.device,
            "verbosity": 0,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostForecastBank":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or y.ndim != 2 or y.shape[1] != self.output_len:
            raise ValueError(f"expected X=(n,p), y=(n,{self.output_len})")

        self.point_models = []
        self.quantile_models = []
        for horizon in range(self.output_len):
            point_model = XGBRegressor(
                objective="reg:squarederror", **self._common_params()
            )
            quantile_model = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=self.quantiles.astype(np.float64),
                **self._common_params(),
            )
            point_model.fit(X, y[:, horizon])
            quantile_model.fit(X, y[:, horizon])
            self.point_models.append(point_model)
            self.quantile_models.append(quantile_model)
        return self

    def _check_fitted(self) -> None:
        if len(self.point_models) != self.output_len:
            raise RuntimeError("forecast bank has not been fitted")

    def predict_point(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.column_stack(
            [model.predict(X) for model in self.point_models]
        ).astype(np.float32)

    def predict_quantiles(
        self,
        X: np.ndarray,
        enforce_monotonicity: bool = True,
    ) -> np.ndarray:
        """Return quantile forecasts as ``(windows, horizon, quantiles)``."""

        self._check_fitted()
        horizon_predictions = []
        for model in self.quantile_models:
            prediction = np.asarray(model.predict(X), dtype=np.float32)
            if prediction.ndim == 1:
                prediction = prediction[:, None]
            horizon_predictions.append(prediction)
        result = np.stack(horizon_predictions, axis=1)
        if enforce_monotonicity:
            result = np.maximum.accumulate(result, axis=2)
        return result.astype(np.float32)
