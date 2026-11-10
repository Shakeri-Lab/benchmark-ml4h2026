"""Isolated one-step autoregression with known insulin and recursive rollout.

This is an ARX / distributed-lag regression, NOT statsmodels SARIMAX. It has
no MA error term. Nothing in the repository's models package is changed.

Timestamp convention: glucose G[t] is predicted from earlier glucose and
the known insulin recorded at t and earlier times. The future insulin plan
is known in arm 2; future observed glucose is never an input to rollout.
For difference_order=1, predict delta G using lagged delta G and insulin,
then integrate the predicted increment once. No clipping is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


MAX_LAG = 48


@dataclass(frozen=True)
class Specification:
    glucose_lags: int
    insulin_lags: int
    difference_order: int
    ridge: float

    def indices(self):
        off = 0 if self.difference_order == 0 else MAX_LAG
        return np.r_[np.arange(off, off + self.glucose_lags),
                     np.arange(2 * MAX_LAG, 2 * MAX_LAG + self.insulin_lags),
                     np.arange(3 * MAX_LAG, 3 * MAX_LAG + self.insulin_lags)]


def design_rows(values, target_indices):
    """Shared, train-only design: levels, differences, basal and bolus lags.

    values columns = glucose, basal, bolus; target_indices are within ONE
    contiguous patient/sequence. Returns all lags for sufficient statistics.
    """
    a = np.asarray(values, dtype=np.float64)
    t = np.asarray(target_indices, dtype=int)
    if t.size == 0 or t.min() <= MAX_LAG or t.max() >= len(a):
        raise ValueError("Insufficient observed history or invalid target index")
    lag = np.arange(1, MAX_LAG + 1)
    glucose = a[t[:, None] - lag, 0]
    differences = glucose - a[t[:, None] - lag - 1, 0]
    insulin_t = t[:, None] - np.arange(MAX_LAG)
    x = np.column_stack((glucose, differences,
                         a[insulin_t, 1], a[insulin_t, 2]))
    y = np.column_stack((a[t, 0], a[t, 0] - a[t - 1, 0]))
    return x, y


def fit_moments(spec, moments):
    """Ridge objective: mean squared error + lambda * standardized beta^2.

    Intercept is unpenalized. All scaling and sufficient statistics use
    training observations only. OLS uses a symmetric pseudoinverse.
    """
    idx = spec.indices()
    n = float(moments["n"])
    mx = moments["sum_x"][idx] / n
    my = moments["sum_y"][spec.difference_order] / n
    covariance = moments["xtx"][np.ix_(idx, idx)] / n - np.outer(mx, mx)
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    scale = np.where(scale > 1e-10, scale, 1.0)
    gram = covariance / np.outer(scale, scale)
    rhs = (moments["xty"][idx, spec.difference_order] / n - mx * my) / scale
    if spec.ridge == 0:
        standardized_coef = np.linalg.pinv(gram, rcond=1e-10, hermitian=True) @ rhs
    else:
        standardized_coef = np.linalg.solve(gram + spec.ridge * np.eye(len(idx)), rhs)
    coef = standardized_coef / scale
    return RecursiveARX(spec, coef, float(my - mx @ coef))


class RecursiveARX:
    def __init__(self, spec, coefficients, intercept):
        self.spec = spec
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.intercept = float(intercept)
        if len(self.coefficients) != spec.glucose_lags + 2 * spec.insulin_lags:
            raise ValueError("Coefficient/design width mismatch")

    def spectral_radius(self):
        p = self.spec.glucose_lags
        companion = np.zeros((p, p))
        companion[0] = self.coefficients[:p]
        if p > 1:
            companion[1:, :-1] = np.eye(p - 1)
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    def rollout(self, observed_history, future_insulin):
        """Batch rollout. The signature intentionally accepts NO future glucose."""
        history = np.asarray(observed_history, dtype=np.float64)
        plan = np.asarray(future_insulin, dtype=np.float64)
        if history.ndim == 2:
            history = history[None]
        if plan.ndim == 2:
            plan = plan[None]
        b, h, c = history.shape
        if c != 3 or plan.shape[0] != b or plan.shape[2] != 2:
            raise ValueError("Expected history [B,H,3] and insulin plan [B,L,2]")
        p, q = self.spec.glucose_lags, self.spec.insulin_lags
        if h < max(p + self.spec.difference_order, q - 1):
            raise ValueError("Insufficient observed history")
        if not np.isfinite(history).all() or not np.isfinite(plan).all():
            raise ValueError("Nonfinite inputs")
        horizon = plan.shape[1]
        glucose = np.empty((b, h + horizon), dtype=np.float64)
        glucose[:, :h] = history[:, :, 0]
        insulin = np.concatenate((history[:, :, 1:], plan), axis=1)
        for k in range(horizon):
            t = h + k
            ar = glucose[:, t - np.arange(1, p + 1)]
            if self.spec.difference_order:
                ar = ar - glucose[:, t - np.arange(2, p + 2)]
            known = insulin[:, t - np.arange(q), :]
            x = np.column_stack((ar, known[:, :, 0], known[:, :, 1]))
            prediction = self.intercept + x @ self.coefficients
            if self.spec.difference_order:
                prediction += glucose[:, t - 1]
            glucose[:, t] = prediction
        return glucose[:, h:]
