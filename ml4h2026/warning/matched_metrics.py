"""Bootstrap helpers for descriptive test-rate-matched operating curves."""
import numpy as np


def choose_indices(fa, auc, recall0, cutoffs, target):
    """Vectorized equivalent of the declared closest-under-target selection rule."""
    fa = np.atleast_2d(fa)
    auc = np.broadcast_to(np.atleast_2d(auc), fa.shape)
    recall0 = np.broadcast_to(np.atleast_2d(recall0), fa.shape)
    eligible = fa <= target + 1e-12
    if not eligible.any(axis=1).all():
        raise RuntimeError("The no-alarm candidate must always be feasible.")
    highest = np.where(eligible, fa, -np.inf).max(axis=1)
    ties = eligible & (fa == highest[:, None])
    area = np.where(np.isfinite(auc), auc, -1)
    best_auc = np.where(ties, area, -np.inf).max(axis=1)
    ties &= area == best_auc[:, None]
    rec = np.where(np.isfinite(recall0), recall0, -1)
    best_rec = np.where(ties, rec, -np.inf).max(axis=1)
    ties &= rec == best_rec[:, None]
    # None = no-alarm candidate, lower than every numeric cutoff.
    numeric = np.asarray([-1e100 if c is None else c for c in cutoffs])
    return np.argmin(np.where(ties, numeric[None, :], np.inf), axis=1)


def bootstrap_weights(n_patients, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_patients, (n_boot, n_patients))
    return np.asarray([np.bincount(row, minlength=n_patients) for row in indices], dtype=float)


def weighted_selected_curves(recall, chosen, weights):
    """Recall array: thresholds x patients x lead; reselect threshold per draw."""
    selected = recall[chosen]
    finite = np.isfinite(selected)
    numerator = np.sum(weights[:, :, None] * np.nan_to_num(selected, nan=0.0), axis=1)
    denominator = np.sum(weights[:, :, None] * finite, axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def bootstrap_selection(fa, overall_recall, cutoffs, targets, leads, weights):
    """Bootstrap patients, then recompute the operating cutoff inside each draw."""
    fa_boot = weights @ fa.T / weights.sum(axis=1, keepdims=True)
    supported = np.isfinite(overall_recall[0, :, 0])
    denom = weights @ supported.astype(float)
    if hasattr(np, "trapezoid"):
        auc_patient = np.trapezoid(overall_recall, leads, axis=2) / 60
    else:
        auc_patient = np.trapz(overall_recall, leads, axis=2) / 60
    def mean_boot(values):
        num = weights @ np.nan_to_num(values, nan=0.0).T
        return np.divide(num, denom[:, None], out=np.full_like(num, np.nan), where=denom[:, None] > 0)
    auc_boot = mean_boot(auc_patient)
    rec0_boot = mean_boot(overall_recall[:, :, 0])
    return {target: choose_indices(fa_boot, auc_boot, rec0_boot, cutoffs, target) for target in targets}
