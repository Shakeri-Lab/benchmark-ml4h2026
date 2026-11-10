"""Three-input glucose forecaster; no hypoglycemia labels enter model fitting.

IOB remaining fraction follows Lee et al. (2009), Eq. (1), tau=100 min:
https://doi.org/10.1177/193229680900300511
The glucose mapping is a deliberately simple fitted linear baseline, not that
paper's MPC algorithm and not a clinically validated dosing model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FEATURES = ("last_cgm_mgdl", "last_cgm_derivative_mgdl_per_min", "iob_units")


def remaining_fraction(age_minutes, tau_minutes=100.0):
    age = np.asarray(age_minutes, dtype=float)
    if not np.isfinite(tau_minutes) or tau_minutes <= 0:
        raise ValueError("IOB time constant must be finite and positive.")
    if not np.isfinite(age).all() or (age < 0).any():
        raise ValueError("Dose ages must be finite and nonnegative.")
    ratio = age / tau_minutes
    return (1 + ratio) * np.exp(-ratio)


def insulin_on_board(slots, basal_u_per_hour, bolus_units, *,
                     tau_minutes=100.0, history_steps=288, step_minutes=5.0):
    """Causal convolution, reset at gaps, with a fixed finite history.

    Basal rate samples are converted to U per five-minute bin and approximated
    as point doses at the recorded bin timestamp. Recorded standard/extended
    bolus amounts must be delivered units, not rates or future dose plans.
    No normal-basal subtraction is used: this is total rapid-acting IOB.
    """
    slots = np.asarray(slots, dtype=np.int64)
    basal = np.asarray(basal_u_per_hour, dtype=float)
    bolus = np.asarray(bolus_units, dtype=float)
    if slots.ndim != 1 or basal.shape != slots.shape or bolus.shape != slots.shape:
        raise ValueError("Expected equally sized one-dimensional arrays.")
    if history_steps < 1 or step_minutes <= 0:
        raise ValueError("Invalid history or sampling interval.")
    if not np.isfinite(basal).all() or not np.isfinite(bolus).all():
        raise ValueError("Missing/nonfinite insulin cannot be silently treated as zero.")
    if (basal < 0).any() or (bolus < 0).any() or (np.diff(slots) <= 0).any():
        raise ValueError("Negative insulin or unordered timestamps.")
    result = np.empty(len(slots), dtype=float)
    kernel = remaining_fraction(np.arange(history_steps) * step_minutes, tau_minutes)
    boundaries = np.r_[0, np.flatnonzero(np.diff(slots) != 1) + 1, len(slots)]
    dose = basal * step_minutes / 60.0 + bolus
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if hi > lo:
            result[lo:hi] = np.convolve(dose[lo:hi], kernel, mode="full")[:hi-lo]
    return result


def features_from_sequence(cgm, slots, basal_u_per_hour, bolus_units, **kwargs):
    cgm = np.asarray(cgm, dtype=float)
    slots = np.asarray(slots, dtype=np.int64)
    if cgm.shape != slots.shape or not np.isfinite(cgm).all():
        raise ValueError("Finite CGM and equally sized timestamps are required.")
    slope = np.full(len(cgm), np.nan)
    adjacent = np.flatnonzero(np.diff(slots) == 1) + 1
    slope[adjacent] = (cgm[adjacent] - cgm[adjacent-1]) / kwargs.get("step_minutes", 5.0)
    iob = insulin_on_board(slots, basal_u_per_hour, bolus_units, **kwargs)
    return np.column_stack((cgm, slope, iob))


def valid_origins(slots, history_steps=288, horizon_steps=12, stride=1):
    """Never bridge gaps in the shared history or future target window."""
    slots = np.asarray(slots, dtype=np.int64)
    if (np.diff(slots) <= 0).any() or min(history_steps, horizon_steps, stride) < 1:
        raise ValueError("Invalid timestamps or window specification.")
    origins = np.arange(history_steps-1, max(history_steps-1, len(slots)-horizon_steps), stride)
    keep = ((slots[origins] - slots[origins-history_steps+1] == history_steps-1) &
            (slots[origins+horizon_steps] - slots[origins] == horizon_steps))
    return origins[keep]


@dataclass
class IOBLinearForecaster:
    """Direct multi-output ridge prediction of glucose change at every lead.

    predict(features) returns glucose values, not an event probability/score.
    Means/scales are learned from training features only. All outputs have an
    unpenalized intercept and exactly three (unconstrained) feature coefficients.
    """
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    alpha: float

    @classmethod
    def fit(cls, features, future_glucose, alpha=0.0):
        x = np.asarray(features, dtype=float)
        y = np.asarray(future_glucose, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or y.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("Fit requires three inputs and a future glucose trajectory.")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha < 0:
            raise ValueError("Nonfinite fitting data or negative regularization.")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        z = np.column_stack((np.ones(len(x)), (x-mean)/scale))
        change = y - x[:, :1]
        gram = z.T @ z / len(z)
        penalty = np.diag([0.0, alpha, alpha, alpha])
        coefficients = np.linalg.lstsq(gram+penalty, z.T @ change / len(z), rcond=None)[0]
        return cls(mean, scale, coefficients, float(alpha))

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError("Prediction requires finite three-feature rows.")
        z = np.column_stack((np.ones(len(x)), (x-self.mean)/self.scale))
        result = x[:, :1] + z @ self.coefficients
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite glucose forecast.")
        return result

    def to_dict(self):
        return dict(feature_names=FEATURES, mean=self.mean.tolist(), scale=self.scale.tolist(),
                    coefficients=self.coefficients.tolist(), alpha=self.alpha,
                    output="glucose_mgdl_at_each_future_five_minute_step",
                    fit_target="future_glucose_minus_last_glucose", insulin_sign_constrained=False)

    @classmethod
    def from_dict(cls, document):
        if tuple(document["feature_names"]) != FEATURES:
            raise ValueError("Unknown feature order.")
        return cls(np.asarray(document["mean"]), np.asarray(document["scale"]),
                   np.asarray(document["coefficients"]), float(document["alpha"]))
