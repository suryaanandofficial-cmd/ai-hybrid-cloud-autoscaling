"""
Forecasting and replica helpers for the predictive autoscaling prototype.

Only needs numpy, so it runs anywhere the dashboard runs.
"""
import math

import numpy as np


# ------------------------------------------------------------------
# Replica recommendation (same rule as the first prototype)
# ------------------------------------------------------------------

def recommend_replicas(request_rate, min_replicas, max_replicas, requests_per_replica):
    """Replicas needed for `request_rate` (requests per second).

    `requests_per_replica` is requests per MINUTE one replica can handle,
    the same meaning as the dashboard setting. The answer is always kept
    between min_replicas and max_replicas.
    """
    if min_replicas > max_replicas:
        raise ValueError("min_replicas must not exceed max_replicas")
    if requests_per_replica <= 0:
        raise ValueError("requests_per_replica must be positive")
    per_minute = max(0.0, float(request_rate)) * 60
    needed = math.ceil(round(per_minute / requests_per_replica, 9))
    return max(min_replicas, min(needed, max_replicas))


# ------------------------------------------------------------------
# Windows and a linear forecaster
# ------------------------------------------------------------------

def make_windows(values, lookback, horizon=1):
    """Turn a series into (inputs, targets).

    Each input is `lookback` consecutive values; its target is the value
    `horizon` steps after the window ends.
    """
    values = np.asarray(values, dtype=float)
    n = len(values) - lookback - horizon + 1
    if n <= 0:
        return np.empty((0, lookback)), np.empty(0)
    X = np.stack([values[i:i + lookback] for i in range(n)])
    y = values[lookback + horizon - 1: lookback + horizon - 1 + n]
    return X, y


def fit_linear(X, y):
    """Least-squares weights (last entry is the intercept)."""
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predict_linear(coef, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ coef


class LinearForecaster:
    """Predicts the value `horizon` steps ahead from the last `lookback` values."""

    def __init__(self, lookback=12, horizon=1):
        self.lookback = lookback
        self.horizon = horizon
        self.coef = None

    def fit(self, series):
        X, y = make_windows(series, self.lookback, self.horizon)
        if len(y) < 2 * (self.lookback + 1):
            raise ValueError(
                f"need at least {2 * (self.lookback + 1)} training windows, got {len(y)}"
            )
        self.coef = fit_linear(X, y)
        return self

    def predict_next(self, series):
        window = np.asarray(series, dtype=float)[-self.lookback:]
        if len(window) < self.lookback:
            raise ValueError("series shorter than the lookback window")
        if self.coef is None:
            raise ValueError("fit the model first")
        return max(0.0, float(predict_linear(self.coef, window[None, :])[0]))


def rolling_forecast(series, lookback, horizon, min_train, last=None):
    """Honest out-of-sample forecasts.

    For each time t the model is fitted ONLY on series[:t], then predicts
    the value `horizon` steps after the last known point. Returns three
    arrays: indices of the forecast targets, predictions, actual values.
    """
    series = np.asarray(series, dtype=float)
    start = min_train
    if last is not None:
        start = max(start, len(series) - horizon + 1 - last)
    idx, pred, actual = [], [], []
    for t in range(start, len(series) - horizon + 1):
        try:
            model = LinearForecaster(lookback, horizon).fit(series[:t])
        except ValueError:
            continue
        idx.append(t + horizon - 1)
        pred.append(model.predict_next(series[:t]))
        actual.append(series[t + horizon - 1])
    return np.array(idx, dtype=int), np.array(pred), np.array(actual)


def rmse(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))
