from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ------------------ Baseline Clarke/Parkes implementations ------------------ #
# Adapted (logic-equivalent) from GARNN error grids implementation:
# Piao et al., "GARNN: An interpretable graph attentive recurrent neural network for predicting blood glucose levels via multivariate time series"
# Source: https://github.com/ChengzhePiao/garnn_public/blob/main/error_grids.py


def _clarke_error_zone_detailed_scalar(act, pred):
    # Zone A
    if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
        return 0
    # Zone E - left upper
    if act <= 70 and pred >= 180:
        return 8
    # Zone E - right lower
    if act >= 180 and pred <= 70:
        return 7
    # Zone D - right
    if act >= 240 and 70 <= pred <= 180:
        return 6
    # Zone D - left
    if act <= 70 <= pred <= 180:
        return 5
    # Zone C - upper
    if 70 <= act <= 290 and pred >= act + 110:
        return 4
    # Zone C - lower
    if 130 <= act <= 180 and pred <= (7 / 5) * act - 182:
        return 3
    # Zone B - upper
    if act < pred:
        return 2
    # Zone B - lower
    return 1


def _parkes_error_zone_detailed_scalar(act, pred, diabetes_type):
    def above_line(x_1, y_1, x_2, y_2, strict=False):
        if x_1 == x_2:
            return False
        y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x_1, y_1, x_2, y_2, strict=False):
        return not above_line(x_1, y_1, x_2, y_2, not strict)

    def parkes_type_1(act, pred):
        # Zone E
        if above_line(0, 150, 35, 155) and above_line(35, 155, 50, 550):
            return 7
        # Zone D - left upper
        if (
            pred > 100
            and above_line(25, 100, 50, 125)
            and above_line(50, 125, 80, 215)
            and above_line(80, 215, 125, 550)
        ):
            return 6
        # Zone D - right lower
        if act > 250 and below_line(250, 40, 550, 150):
            return 5
        # Zone C - left upper
        if (
            pred > 60
            and above_line(30, 60, 50, 80)
            and above_line(50, 80, 70, 110)
            and above_line(70, 110, 260, 550)
        ):
            return 4
        # Zone C - right lower
        if act > 120 and below_line(120, 30, 260, 130) and below_line(260, 130, 550, 250):
            return 3
        # Zone B - left upper
        if (
            pred > 50
            and above_line(30, 50, 140, 170)
            and above_line(140, 170, 280, 380)
            and (act < 280 or above_line(280, 380, 430, 550))
        ):
            return 2
        # Zone B - right lower
        if (
            act > 50
            and below_line(50, 30, 170, 145)
            and below_line(170, 145, 385, 300)
            and (act < 385 or below_line(385, 300, 550, 450))
        ):
            return 1
        # Zone A
        return 0

    def parkes_type_2(act, pred):
        # Zone E
        if pred > 200 and above_line(35, 200, 50, 550):
            return 7
        # Zone D - left upper
        if pred > 80 and above_line(25, 80, 35, 90) and above_line(35, 90, 125, 550):
            return 6
        # Zone D - right lower
        if act > 250 and below_line(250, 40, 410, 110) and below_line(410, 110, 550, 160):
            return 5
        # Zone C - left upper
        if pred > 60 and above_line(30, 60, 280, 550):
            return 4
        # Zone C - right lower
        if below_line(90, 0, 260, 130) and below_line(260, 130, 550, 250):
            return 3
        # Zone B - left upper
        if pred > 50 and above_line(30, 50, 230, 330) and (act < 230 or above_line(230, 330, 440, 550)):
            return 2
        # Zone B - right lower
        if (
            act > 50
            and below_line(50, 30, 90, 80)
            and below_line(90, 80, 330, 230)
            and (act < 330 or below_line(330, 230, 550, 450))
        ):
            return 1
        # Zone A
        return 0

    if diabetes_type == 1:
        return parkes_type_1(act, pred)
    if diabetes_type == 2:
        return parkes_type_2(act, pred)
    raise Exception("Unsupported diabetes type")


clarke_error_zone_detailed = np.vectorize(_clarke_error_zone_detailed_scalar)
parkes_error_zone_detailed = np.vectorize(_parkes_error_zone_detailed_scalar)


def zone_merge_counts(counts: np.ndarray) -> np.ndarray:
    """
    Merge detailed counts into A-E.
    Clarke: 0 | 1+2 | 3+4 | 5+6 | 7+8
    Parkes: 0 | 1+2 | 3+4 | 5+6 | 7
    """
    merged = np.zeros(5, dtype=float)
    merged[0] = counts[0]
    merged[1] = counts[1] + counts[2]
    merged[2] = counts[3] + counts[4]
    merged[3] = counts[5] + counts[6]
    merged[4] = counts[7] + (counts[8] if counts.shape[0] > 8 else 0)
    return merged


# ---------------------------- Clinical Metrics ----------------------------- #


def _broadcast_pair(y, y_hat):
    y_arr = np.array(y)
    yhat_arr = np.array(y_hat)
    # Align ranks by prepending singleton dims
    while y_arr.ndim < yhat_arr.ndim:
        y_arr = np.expand_dims(y_arr, axis=0)
    while yhat_arr.ndim < y_arr.ndim:
        yhat_arr = np.expand_dims(yhat_arr, axis=0)
    y_b, yhat_b = np.broadcast_arrays(y_arr, yhat_arr)
    return y_b, yhat_b


def _clip_idx(arr, max_idx: int):
    idx = np.rint(arr).astype(np.int64)
    return np.clip(idx, 0, max_idx)


def _clip_glucose(arr, max_val: float):
    return np.clip(arr, 0.0, max_val)


def _sigma_ge(x, a, eps):
    xi = (2.0 / eps) * (x - a) - 1.0
    res = np.zeros_like(x, dtype=float)
    # a <= x <= a + eps/2
    mask1 = (x >= a) & (x <= a + eps / 2)
    res[mask1] = -0.5 * xi[mask1] ** 4 - xi[mask1] ** 3 + xi[mask1] + 0.5
    # a + eps/2 <= x <= a + eps
    mask2 = (x > a + eps / 2) & (x <= a + eps)
    res[mask2] = 0.5 * xi[mask2] ** 4 - xi[mask2] ** 3 + xi[mask2] + 0.5
    res[x > a + eps] = 1.0
    return res


def _sigma_le_bar(x, a, eps):
    xi = (2.0 / eps) * (x - a) + 1.0
    res = np.ones_like(x, dtype=float)
    # a - eps <= x <= a - eps/2
    mask1 = (x >= a - eps) & (x <= a - eps / 2)
    res[mask1] = 0.5 * xi[mask1] ** 4 - xi[mask1] ** 3 + xi[mask1] + 0.5
    # a - eps/2 <= x <= a
    mask2 = (x > a - eps / 2) & (x <= a)
    res[mask2] = -0.5 * xi[mask2] ** 4 - xi[mask2] ** 3 + xi[mask2] + 0.5
    res[x > a] = 0.0
    return res


@dataclass
class ClinicalMetrics:
    risk_matrix: np.ndarray

    @classmethod
    def from_default(cls):
        path = os.path.join(os.path.dirname(__file__), "risk_matrix_seg.npy")
        risk = np.load(path)
        return cls(risk_matrix=risk)

    # -------- Core metrics -------- #
    def SEG(self, y, y_hat, reduction: str = "mean"):
        y_b, yhat_b = _broadcast_pair(y, y_hat)
        max_yhat = int(self.risk_matrix.shape[0] - 1)
        max_y = int(self.risk_matrix.shape[1] - 1)
        yi = _clip_idx(y_b, max_y)
        yhi = _clip_idx(yhat_b, max_yhat)
        risk = self.risk_matrix[yhi, yi]
        if reduction == "none":
            return risk
        if reduction == "sum":
            return float(risk.sum())
        return float(risk.mean())

    def seg_z_zone(self, y, y_hat):
        """SEG zones based on risk: [0-0.5), [0.5-1.5), [1.5-2.5), [2.5-3.5), >=3.5."""
        risk = self.SEG(y, y_hat, reduction="none")
        bins = [0.5, 1.5, 2.5, 3.5]
        return np.digitize(risk, bins=bins, right=False)

    def seg_z_zone_from_risk(self, risk):
        bins = [0.5, 1.5, 2.5, 3.5]
        return np.digitize(risk, bins=bins, right=False)

    def seg_z_ratio(self, y, y_hat):
        zones = self.seg_z_zone(y, y_hat).ravel()
        counts = np.bincount(zones, minlength=5)
        return counts / counts.sum() if counts.sum() > 0 else np.full(5, np.nan)

    def RMSE(self, y, y_hat, reduction: str = "mean"):
        y_b, yhat_b = _broadcast_pair(y, y_hat)
        se = (y_b - yhat_b) ** 2
        if reduction == "none":
            return se
        if reduction == "sum":
            return float(np.sqrt(se.sum()))
        return float(np.sqrt(se.mean()))

    def MAE(self, y, y_hat, reduction: str = "mean"):
        y_b, yhat_b = _broadcast_pair(y, y_hat)
        ae = np.abs(y_b - yhat_b)
        if reduction == "none":
            return ae
        if reduction == "sum":
            return float(ae.sum())
        return float(ae.mean())

    # -------- gMSE / gRMSE -------- #
    def gSE(self, y, y_hat):
        g, gh = _broadcast_pair(y, y_hat)
        se = (g - gh) ** 2
        TL, TH = 85.0, 155.0
        alpha_L, alpha_H = 1.5, 1.0
        beta_L, beta_H = 30.0, 100.0
        gamma_L, gamma_H = 10.0, 20.0

        pen = 1.0
        pen += alpha_L * _sigma_le_bar(g, TL, beta_L) * _sigma_ge(gh, g, gamma_L)
        pen += alpha_H * _sigma_ge(g, TH, beta_H) * _sigma_le_bar(gh, g, gamma_H)
        return se * pen

    def gMSE(self, y, y_hat, reduction: str = "mean"):
        gse = self.gSE(y, y_hat)
        if reduction == "none":
            return gse
        if reduction == "sum":
            return float(gse.sum())
        return float(gse.mean())

    def gRMSE(self, y, y_hat, reduction: str = "mean"):
        gmse = self.gMSE(y, y_hat, reduction=reduction)
        if reduction == "none":
            return np.sqrt(gmse)
        if reduction == "sum":
            return float(np.sqrt(gmse))
        return float(np.sqrt(gmse))

    # -------- CEG / PEG -------- #
    def ceg_zone(self, y, y_hat):
        y_b, yhat_b = _broadcast_pair(y, y_hat)
        y_b = _clip_glucose(y_b, 600.0)
        yhat_b = _clip_glucose(yhat_b, 600.0)
        return clarke_error_zone_detailed(y_b, yhat_b)

    def peg_zone(self, y, y_hat, diabetes_type: int = 1):
        y_b, yhat_b = _broadcast_pair(y, y_hat)
        y_b = _clip_glucose(y_b, 600.0)
        yhat_b = _clip_glucose(yhat_b, 600.0)
        return parkes_error_zone_detailed(y_b, yhat_b, diabetes_type)

    def ceg_ratio(self, y, y_hat, detailed: bool = False):
        zones = self.ceg_zone(y, y_hat).ravel()
        counts = np.bincount(zones, minlength=9)
        if detailed:
            return counts / counts.sum()
        merged = zone_merge_counts(counts)
        return merged / merged.sum()

    def peg_ratio(self, y, y_hat, detailed: bool = False, diabetes_type: int = 1):
        zones = self.peg_zone(y, y_hat, diabetes_type=diabetes_type).ravel()
        counts = np.bincount(zones, minlength=9)
        if detailed:
            return counts / counts.sum()
        merged = zone_merge_counts(counts)
        return merged / merged.sum()


# ----------------------------- Aggregator ---------------------------------- #


ICGM_KEYS = [
    "A_n",
    "A_good",
    "B_n",
    "B_good",
    "C_n",
    "C_good",
    "D_n",
    "D_good",
    "E_n",
    "E_good",
    "F_n",
    "F_good",
    "G_n",
    "G_good",
    "H_count",
    "I_count",
    "J_n",
    "J_bad",
    "K_n",
    "K_bad",
]


def _init_icgm_counts(shape=None):
    if shape is None:
        return {k: 0.0 for k in ICGM_KEYS}
    return {k: np.zeros(shape, dtype=float) for k in ICGM_KEYS}


def _wilson_lcb(k: float, n: float, z: float = 1.6448536269514722) -> float:
    if n <= 0:
        return np.nan
    p = k / n
    denom = 1.0 + (z * z) / n
    center = p + (z * z) / (2.0 * n)
    adj = z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)
    return (center - adj) / denom


def _icgm_from_counts(counts: Dict[str, float]) -> Dict[str, Any]:
    def _frac_lcb(good, n):
        if n <= 0:
            return np.nan, np.nan
        frac = good / n
        lcb = _wilson_lcb(good, n)
        return frac, lcb

    criteria = {}

    frac, lcb = _frac_lcb(counts["A_good"], counts["A_n"])
    criteria["A"] = {"n": counts["A_n"], "frac": frac, "lcb": lcb, "pass": bool(lcb > 0.85) if not np.isnan(lcb) else False}
    frac, lcb = _frac_lcb(counts["B_good"], counts["B_n"])
    criteria["B"] = {"n": counts["B_n"], "frac": frac, "lcb": lcb, "pass": bool(lcb > 0.70) if not np.isnan(lcb) else False}
    frac, lcb = _frac_lcb(counts["C_good"], counts["C_n"])
    criteria["C"] = {"n": counts["C_n"], "frac": frac, "lcb": lcb, "pass": bool(lcb > 0.80) if not np.isnan(lcb) else False}
    frac, lcb = _frac_lcb(counts["D_good"], counts["D_n"])
    criteria["D"] = {"n": counts["D_n"], "frac": frac, "lcb": lcb, "pass": bool(lcb > 0.98) if not np.isnan(lcb) else False}
    frac, lcb = _frac_lcb(counts["E_good"], counts["E_n"])
    criteria["E"] = {"n": counts["E_n"], "frac": frac, "lcb": lcb, "pass": bool(lcb > 0.99) if not np.isnan(lcb) else False}
    frac, lcb = _frac_lcb(counts["F_good"], counts["F_n"])
    criteria["F"] = {"n": counts["F_n"], "frac": frac, "lcb": lcb, "pass": bool(lcb > 0.99) if not np.isnan(lcb) else False}
    frac, lcb = _frac_lcb(counts["G_good"], counts["G_n"])
    criteria["G"] = {"n": counts["G_n"], "frac": frac, "lcb": lcb, "pass": bool(lcb > 0.87) if not np.isnan(lcb) else False}

    criteria["H"] = {"n": counts["H_count"], "pass": bool(counts["H_count"] == 0)}
    criteria["I"] = {"n": counts["I_count"], "pass": bool(counts["I_count"] == 0)}

    # Rate-of-change criteria: pass if bad fraction <= 0.01
    def _roc_pass(bad, n):
        if n <= 0:
            return True, np.nan
        frac = bad / n
        return frac <= 0.01, frac

    pass_j, frac_j = _roc_pass(counts["J_bad"], counts["J_n"])
    pass_k, frac_k = _roc_pass(counts["K_bad"], counts["K_n"])
    criteria["J"] = {"n": counts["J_n"], "bad_frac": frac_j, "pass": bool(pass_j)}
    criteria["K"] = {"n": counts["K_n"], "bad_frac": frac_k, "pass": bool(pass_k)}

    overall_pass = all(v["pass"] for v in criteria.values())
    return {"pass": overall_pass, "criteria": criteria}


class MetricsAggregator:
    def __init__(self, diabetes_type: int = 1):
        self.metrics = ClinicalMetrics.from_default()
        self.diabetes_type = diabetes_type
        self.reset()

    def reset(self):
        self.count = 0
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.sum_seg = 0.0
        self.sum_gse = 0.0
        self.ceg_counts = np.zeros(9, dtype=float)
        self.peg_counts = np.zeros(9, dtype=float)
        self.segz_counts = np.zeros(5, dtype=float)
        self.icgm_counts = _init_icgm_counts()
        # Hypo/hyper classification counts
        self.hypo_tp = 0.0
        self.hypo_fp = 0.0
        self.hypo_fn = 0.0
        self.hyper_tp = 0.0
        self.hyper_fp = 0.0
        self.hyper_fn = 0.0
        self.sev_hypo_tp = 0.0
        self.sev_hypo_fp = 0.0
        self.sev_hypo_fn = 0.0
        self.per_horizon = None
        self.per_patient: Dict[str, Dict[str, float]] = {}

    def _ensure_horizon_arrays(self, H: int):
        if self.per_horizon is None:
            self.per_horizon = {
                "count": np.zeros(H, dtype=float),
                "sum_sq": np.zeros(H, dtype=float),
                "sum_abs": np.zeros(H, dtype=float),
                "sum_seg": np.zeros(H, dtype=float),
                "sum_gse": np.zeros(H, dtype=float),
                "ceg_counts": np.zeros((H, 9), dtype=float),
                "peg_counts": np.zeros((H, 9), dtype=float),
                "segz_counts": np.zeros((H, 5), dtype=float),
                "icgm_counts": _init_icgm_counts(H),
                "hypo_tp": np.zeros(H, dtype=float),
                "hypo_fp": np.zeros(H, dtype=float),
                "hypo_fn": np.zeros(H, dtype=float),
                "hyper_tp": np.zeros(H, dtype=float),
                "hyper_fp": np.zeros(H, dtype=float),
                "hyper_fn": np.zeros(H, dtype=float),
                "sev_hypo_tp": np.zeros(H, dtype=float),
                "sev_hypo_fp": np.zeros(H, dtype=float),
                "sev_hypo_fn": np.zeros(H, dtype=float),
            }
            return
        if self.per_horizon["count"].shape[0] < H:
            pad = H - self.per_horizon["count"].shape[0]
            for k in ["count", "sum_sq", "sum_abs", "sum_seg", "sum_gse"]:
                self.per_horizon[k] = np.pad(self.per_horizon[k], (0, pad))
            for k in ["ceg_counts", "peg_counts"]:
                self.per_horizon[k] = np.pad(self.per_horizon[k], ((0, pad), (0, 0)))
            for k in ["segz_counts"]:
                self.per_horizon[k] = np.pad(self.per_horizon[k], ((0, pad), (0, 0)))
            for k, arr in self.per_horizon["icgm_counts"].items():
                self.per_horizon["icgm_counts"][k] = np.pad(arr, (0, pad))
            for k in [
                "hypo_tp",
                "hypo_fp",
                "hypo_fn",
                "hyper_tp",
                "hyper_fp",
                "hyper_fn",
                "sev_hypo_tp",
                "sev_hypo_fp",
                "sev_hypo_fn",
            ]:
                self.per_horizon[k] = np.pad(self.per_horizon[k], (0, pad))

    def update(self, y, y_hat, subject_id: Optional[Iterable] = None, mask=None):
        y_b, yhat_b = _broadcast_pair(y, y_hat)
        if mask is not None:
            mask_b, _ = _broadcast_pair(mask, y_b)
            valid = mask_b.astype(bool)
        else:
            valid = np.ones_like(y_b, dtype=bool)

        se = (y_b - yhat_b) ** 2
        ae = np.abs(y_b - yhat_b)
        seg = self.metrics.SEG(y_b, yhat_b, reduction="none")
        seg_z = self.metrics.seg_z_zone_from_risk(seg)
        gse = self.metrics.gSE(y_b, yhat_b)
        ceg = self.metrics.ceg_zone(y_b, yhat_b)
        peg = self.metrics.peg_zone(y_b, yhat_b, diabetes_type=self.diabetes_type)
        # Classification masks
        true_hypo = y_b < 70.0
        pred_hypo = yhat_b < 70.0
        true_hyper = y_b > 180.0
        pred_hyper = yhat_b > 180.0
        true_sev_hypo = y_b < 54.0
        pred_sev_hypo = yhat_b < 54.0
        err = np.abs(yhat_b - y_b)
        rel_err = err / np.clip(np.abs(y_b), 1e-6, None)
        pred_low = yhat_b < 70.0
        pred_mid = (yhat_b >= 70.0) & (yhat_b <= 180.0)
        pred_high = yhat_b > 180.0

        # Global
        v = valid
        self.count += v.sum()
        self.sum_sq += float((se[v]).sum())
        self.sum_abs += float((ae[v]).sum())
        self.sum_seg += float((seg[v]).sum())
        self.sum_gse += float((gse[v]).sum())
        self.ceg_counts += np.bincount(ceg[v].ravel(), minlength=9)
        self.peg_counts += np.bincount(peg[v].ravel(), minlength=9)
        self.segz_counts += np.bincount(seg_z[v].ravel(), minlength=5)
        # iCGM counts (global)
        self.icgm_counts["A_n"] += float((v & pred_low).sum())
        self.icgm_counts["A_good"] += float((v & pred_low & (err <= 15.0)).sum())
        self.icgm_counts["B_n"] += float((v & pred_mid).sum())
        self.icgm_counts["B_good"] += float((v & pred_mid & (rel_err <= 0.15)).sum())
        self.icgm_counts["C_n"] += float((v & pred_high).sum())
        self.icgm_counts["C_good"] += float((v & pred_high & (rel_err <= 0.15)).sum())
        self.icgm_counts["D_n"] += float((v & pred_low).sum())
        self.icgm_counts["D_good"] += float((v & pred_low & (err <= 40.0)).sum())
        self.icgm_counts["E_n"] += float((v & pred_mid).sum())
        self.icgm_counts["E_good"] += float((v & pred_mid & (rel_err <= 0.40)).sum())
        self.icgm_counts["F_n"] += float((v & pred_high).sum())
        self.icgm_counts["F_good"] += float((v & pred_high & (rel_err <= 0.40)).sum())
        self.icgm_counts["G_n"] += float(v.sum())
        self.icgm_counts["G_good"] += float((v & (rel_err <= 0.20)).sum())
        self.icgm_counts["H_count"] += float((v & pred_low & (y_b > 180.0)).sum())
        self.icgm_counts["I_count"] += float((v & pred_high & (y_b < 70.0)).sum())
        if y_b.ndim >= 2 and y_b.shape[1] > 1:
            pred_rate = np.diff(yhat_b, axis=1) / 5.0
            ref_rate = np.diff(y_b, axis=1) / 5.0
            valid_rate = v[:, 1:, ...] & v[:, :-1, ...]
            j_bad = (pred_rate > 1.0) & (ref_rate < -2.0)
            k_bad = (pred_rate < -1.0) & (ref_rate > 2.0)
            self.icgm_counts["J_n"] += float(valid_rate.sum())
            self.icgm_counts["J_bad"] += float((valid_rate & j_bad).sum())
            self.icgm_counts["K_n"] += float(valid_rate.sum())
            self.icgm_counts["K_bad"] += float((valid_rate & k_bad).sum())
        self.hypo_tp += float((pred_hypo & true_hypo & v).sum())
        self.hypo_fp += float((pred_hypo & ~true_hypo & v).sum())
        self.hypo_fn += float((~pred_hypo & true_hypo & v).sum())
        self.hyper_tp += float((pred_hyper & true_hyper & v).sum())
        self.hyper_fp += float((pred_hyper & ~true_hyper & v).sum())
        self.hyper_fn += float((~pred_hyper & true_hyper & v).sum())
        self.sev_hypo_tp += float((pred_sev_hypo & true_sev_hypo & v).sum())
        self.sev_hypo_fp += float((pred_sev_hypo & ~true_sev_hypo & v).sum())
        self.sev_hypo_fn += float((~pred_sev_hypo & true_sev_hypo & v).sum())

        # Per-horizon (axis=1 if present)
        if y_b.ndim >= 2:
            H = y_b.shape[1]
            self._ensure_horizon_arrays(H)
            for h in range(H):
                v_h = v[:, h, ...] if y_b.ndim > 2 else v[:, h]
                if not np.any(v_h):
                    continue
                se_h = se[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                ae_h = ae[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                seg_h = seg[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                gse_h = gse[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                ceg_h = ceg[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                peg_h = peg[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                segz_h = seg_z[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                v_flat = v_h.ravel()
                self.per_horizon["count"][h] += v_flat.sum()
                self.per_horizon["sum_sq"][h] += se_h.ravel()[v_flat].sum()
                self.per_horizon["sum_abs"][h] += ae_h.ravel()[v_flat].sum()
                self.per_horizon["sum_seg"][h] += seg_h.ravel()[v_flat].sum()
                self.per_horizon["sum_gse"][h] += gse_h.ravel()[v_flat].sum()
                self.per_horizon["ceg_counts"][h] += np.bincount(ceg_h.ravel()[v_flat], minlength=9)
                self.per_horizon["peg_counts"][h] += np.bincount(peg_h.ravel()[v_flat], minlength=9)
                self.per_horizon["segz_counts"][h] += np.bincount(segz_h.ravel()[v_flat], minlength=5)
                # iCGM counts per horizon
                pred_low_h = pred_low[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                pred_mid_h = pred_mid[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                pred_high_h = pred_high[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                err_h = err[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                rel_err_h = rel_err[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                ic = self.per_horizon["icgm_counts"]
                ic["A_n"][h] += float((pred_low_h & v_h).sum())
                ic["A_good"][h] += float((pred_low_h & v_h & (err_h <= 15.0)).sum())
                ic["B_n"][h] += float((pred_mid_h & v_h).sum())
                ic["B_good"][h] += float((pred_mid_h & v_h & (rel_err_h <= 0.15)).sum())
                ic["C_n"][h] += float((pred_high_h & v_h).sum())
                ic["C_good"][h] += float((pred_high_h & v_h & (rel_err_h <= 0.15)).sum())
                ic["D_n"][h] += float((pred_low_h & v_h).sum())
                ic["D_good"][h] += float((pred_low_h & v_h & (err_h <= 40.0)).sum())
                ic["E_n"][h] += float((pred_mid_h & v_h).sum())
                ic["E_good"][h] += float((pred_mid_h & v_h & (rel_err_h <= 0.40)).sum())
                ic["F_n"][h] += float((pred_high_h & v_h).sum())
                ic["F_good"][h] += float((pred_high_h & v_h & (rel_err_h <= 0.40)).sum())
                ic["G_n"][h] += float(v_h.sum())
                ic["G_good"][h] += float((v_h & (rel_err_h <= 0.20)).sum())
                ic["H_count"][h] += float((v_h & pred_low_h & (y_b[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))] > 180.0)).sum())
                ic["I_count"][h] += float((v_h & pred_high_h & (y_b[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))] < 70.0)).sum())
                if y_b.ndim >= 2 and h >= 1:
                    pred_rate_h = np.diff(yhat_b, axis=1)[:, h - 1, ...] / 5.0
                    ref_rate_h = np.diff(y_b, axis=1)[:, h - 1, ...] / 5.0
                    valid_rate_h = (v[:, h, ...] & v[:, h - 1, ...])
                    ic["J_n"][h] += float(valid_rate_h.sum())
                    ic["J_bad"][h] += float((valid_rate_h & (pred_rate_h > 1.0) & (ref_rate_h < -2.0)).sum())
                    ic["K_n"][h] += float(valid_rate_h.sum())
                    ic["K_bad"][h] += float((valid_rate_h & (pred_rate_h < -1.0) & (ref_rate_h > 2.0)).sum())
                true_hypo_h = true_hypo[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                pred_hypo_h = pred_hypo[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                true_hyper_h = true_hyper[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                pred_hyper_h = pred_hyper[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                true_sev_hypo_h = true_sev_hypo[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                pred_sev_hypo_h = pred_sev_hypo[(slice(None), h, *([slice(None)] * (y_b.ndim - 2)))]
                self.per_horizon["hypo_tp"][h] += float((pred_hypo_h & true_hypo_h).ravel()[v_flat].sum())
                self.per_horizon["hypo_fp"][h] += float((pred_hypo_h & ~true_hypo_h).ravel()[v_flat].sum())
                self.per_horizon["hypo_fn"][h] += float((~pred_hypo_h & true_hypo_h).ravel()[v_flat].sum())
                self.per_horizon["hyper_tp"][h] += float((pred_hyper_h & true_hyper_h).ravel()[v_flat].sum())
                self.per_horizon["hyper_fp"][h] += float((pred_hyper_h & ~true_hyper_h).ravel()[v_flat].sum())
                self.per_horizon["hyper_fn"][h] += float((~pred_hyper_h & true_hyper_h).ravel()[v_flat].sum())
                self.per_horizon["sev_hypo_tp"][h] += float((pred_sev_hypo_h & true_sev_hypo_h).ravel()[v_flat].sum())
                self.per_horizon["sev_hypo_fp"][h] += float((pred_sev_hypo_h & ~true_sev_hypo_h).ravel()[v_flat].sum())
                self.per_horizon["sev_hypo_fn"][h] += float((~pred_sev_hypo_h & true_sev_hypo_h).ravel()[v_flat].sum())

        # Per-patient
        if subject_id is not None:
            ids = np.array(subject_id)
            # Expand trailing singleton dims to match y_b
            while ids.ndim < y_b.ndim:
                ids = np.expand_dims(ids, axis=-1)
            ids = np.broadcast_to(ids, y_b.shape).astype(str)
            for pid in np.unique(ids):
                pid_mask = v & (ids == pid)
                if not pid_mask.any():
                    continue
                if pid not in self.per_patient:
                    self.per_patient[pid] = {
                        "count": 0,
                        "sum_sq": 0.0,
                        "sum_abs": 0.0,
                        "sum_seg": 0.0,
                        "sum_gse": 0.0,
                        "ceg_counts": np.zeros(9, dtype=float),
                        "peg_counts": np.zeros(9, dtype=float),
                        "segz_counts": np.zeros(5, dtype=float),
                        "icgm_counts": _init_icgm_counts(),
                        "hypo_tp": 0.0,
                        "hypo_fp": 0.0,
                        "hypo_fn": 0.0,
                        "hyper_tp": 0.0,
                        "hyper_fp": 0.0,
                        "hyper_fn": 0.0,
                        "sev_hypo_tp": 0.0,
                        "sev_hypo_fp": 0.0,
                        "sev_hypo_fn": 0.0,
                    }
                entry = self.per_patient[pid]
                entry["count"] += pid_mask.sum()
                entry["sum_sq"] += float(se[pid_mask].sum())
                entry["sum_abs"] += float(ae[pid_mask].sum())
                entry["sum_seg"] += float(seg[pid_mask].sum())
                entry["sum_gse"] += float(gse[pid_mask].sum())
                entry["ceg_counts"] += np.bincount(ceg[pid_mask].ravel(), minlength=9)
                entry["peg_counts"] += np.bincount(peg[pid_mask].ravel(), minlength=9)
                entry["segz_counts"] += np.bincount(seg_z[pid_mask].ravel(), minlength=5)
                ic = entry["icgm_counts"]
                pid_v = v & pid_mask
                ic["A_n"] += float((pid_v & pred_low).sum())
                ic["A_good"] += float((pid_v & pred_low & (err <= 15.0)).sum())
                ic["B_n"] += float((pid_v & pred_mid).sum())
                ic["B_good"] += float((pid_v & pred_mid & (rel_err <= 0.15)).sum())
                ic["C_n"] += float((pid_v & pred_high).sum())
                ic["C_good"] += float((pid_v & pred_high & (rel_err <= 0.15)).sum())
                ic["D_n"] += float((pid_v & pred_low).sum())
                ic["D_good"] += float((pid_v & pred_low & (err <= 40.0)).sum())
                ic["E_n"] += float((pid_v & pred_mid).sum())
                ic["E_good"] += float((pid_v & pred_mid & (rel_err <= 0.40)).sum())
                ic["F_n"] += float((pid_v & pred_high).sum())
                ic["F_good"] += float((pid_v & pred_high & (rel_err <= 0.40)).sum())
                ic["G_n"] += float(pid_v.sum())
                ic["G_good"] += float((pid_v & (rel_err <= 0.20)).sum())
                ic["H_count"] += float((pid_v & pred_low & (y_b > 180.0)).sum())
                ic["I_count"] += float((pid_v & pred_high & (y_b < 70.0)).sum())
                if y_b.ndim >= 2 and y_b.shape[1] > 1:
                    pred_rate = np.diff(yhat_b, axis=1) / 5.0
                    ref_rate = np.diff(y_b, axis=1) / 5.0
                    valid_rate = pid_v[:, 1:, ...] & pid_v[:, :-1, ...]
                    ic["J_n"] += float(valid_rate.sum())
                    ic["J_bad"] += float((valid_rate & (pred_rate > 1.0) & (ref_rate < -2.0)).sum())
                    ic["K_n"] += float(valid_rate.sum())
                    ic["K_bad"] += float((valid_rate & (pred_rate < -1.0) & (ref_rate > 2.0)).sum())
                entry["hypo_tp"] += float((pred_hypo & true_hypo & pid_mask).sum())
                entry["hypo_fp"] += float((pred_hypo & ~true_hypo & pid_mask).sum())
                entry["hypo_fn"] += float((~pred_hypo & true_hypo & pid_mask).sum())
                entry["hyper_tp"] += float((pred_hyper & true_hyper & pid_mask).sum())
                entry["hyper_fp"] += float((pred_hyper & ~true_hyper & pid_mask).sum())
                entry["hyper_fn"] += float((~pred_hyper & true_hyper & pid_mask).sum())
                entry["sev_hypo_tp"] += float((pred_sev_hypo & true_sev_hypo & pid_mask).sum())
                entry["sev_hypo_fp"] += float((pred_sev_hypo & ~true_sev_hypo & pid_mask).sum())
                entry["sev_hypo_fn"] += float((~pred_sev_hypo & true_sev_hypo & pid_mask).sum())

    def _finalize_counts(
        self,
        count,
        sum_sq,
        sum_abs,
        sum_seg,
        sum_gse,
        ceg_counts,
        peg_counts,
        segz_counts,
        hypo_tp=0.0,
        hypo_fp=0.0,
        hypo_fn=0.0,
        hyper_tp=0.0,
        hyper_fp=0.0,
        hyper_fn=0.0,
        sev_hypo_tp=0.0,
        sev_hypo_fp=0.0,
        sev_hypo_fn=0.0,
        icgm_counts: Dict[str, float] | None = None,
    ):
        if count == 0:
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "seg": np.nan,
                "gmse": np.nan,
                "grmse": np.nan,
                "ceg_ratio": np.full(5, np.nan),
                "peg_ratio": np.full(5, np.nan),
                "seg_z_ratio": np.full(5, np.nan),
                "hypo_recall": np.nan,
                "hypo_precision": np.nan,
                "hyper_recall": np.nan,
                "hyper_precision": np.nan,
                "severe_hypo_recall": np.nan,
                "severe_hypo_precision": np.nan,
                "icgm_pass": False,
                "icgm_criteria": {},
            }
        if icgm_counts is None:
            icgm_counts = _init_icgm_counts()
        icgm_result = _icgm_from_counts(icgm_counts)
        rmse = np.sqrt(sum_sq / count)
        mae = sum_abs / count
        seg = sum_seg / count
        gmse = sum_gse / count
        grmse = np.sqrt(gmse)
        ceg_r = zone_merge_counts(ceg_counts)
        peg_r = zone_merge_counts(peg_counts)
        ceg_r = ceg_r / ceg_r.sum() if ceg_r.sum() > 0 else np.full_like(ceg_r, np.nan)
        peg_r = peg_r / peg_r.sum() if peg_r.sum() > 0 else np.full_like(peg_r, np.nan)
        segz_r = segz_counts / segz_counts.sum() if segz_counts.sum() > 0 else np.full(5, np.nan)
        hypo_recall = hypo_tp / (hypo_tp + hypo_fn) if (hypo_tp + hypo_fn) > 0 else np.nan
        hypo_precision = hypo_tp / (hypo_tp + hypo_fp) if (hypo_tp + hypo_fp) > 0 else np.nan
        hyper_recall = hyper_tp / (hyper_tp + hyper_fn) if (hyper_tp + hyper_fn) > 0 else np.nan
        hyper_precision = hyper_tp / (hyper_tp + hyper_fp) if (hyper_tp + hyper_fp) > 0 else np.nan
        sev_hypo_recall = sev_hypo_tp / (sev_hypo_tp + sev_hypo_fn) if (sev_hypo_tp + sev_hypo_fn) > 0 else np.nan
        sev_hypo_precision = sev_hypo_tp / (sev_hypo_tp + sev_hypo_fp) if (sev_hypo_tp + sev_hypo_fp) > 0 else np.nan
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "seg": float(seg),
            "gmse": float(gmse),
            "grmse": float(grmse),
            "ceg_ratio": ceg_r,
            "peg_ratio": peg_r,
            "seg_z_ratio": segz_r,
            "hypo_recall": float(hypo_recall),
            "hypo_precision": float(hypo_precision),
            "hyper_recall": float(hyper_recall),
            "hyper_precision": float(hyper_precision),
            "severe_hypo_recall": float(sev_hypo_recall),
            "severe_hypo_precision": float(sev_hypo_precision),
            "icgm_pass": bool(icgm_result["pass"]),
            "icgm_criteria": icgm_result["criteria"],
        }

    def compute(self):
        result = {
            "global": self._finalize_counts(
                self.count,
                self.sum_sq,
                self.sum_abs,
                self.sum_seg,
                self.sum_gse,
                self.ceg_counts,
                self.peg_counts,
                self.segz_counts,
                self.hypo_tp,
                self.hypo_fp,
                self.hypo_fn,
                self.hyper_tp,
                self.hyper_fp,
                self.hyper_fn,
                self.sev_hypo_tp,
                self.sev_hypo_fp,
                self.sev_hypo_fn,
                self.icgm_counts,
            )
        }
        if self.per_horizon is not None:
            H = self.per_horizon["count"].shape[0]
            ph = []
            for h in range(H):
                icgm_h = {k: self.per_horizon["icgm_counts"][k][h] for k in ICGM_KEYS}
                ph.append(
                    self._finalize_counts(
                        self.per_horizon["count"][h],
                        self.per_horizon["sum_sq"][h],
                        self.per_horizon["sum_abs"][h],
                        self.per_horizon["sum_seg"][h],
                        self.per_horizon["sum_gse"][h],
                        self.per_horizon["ceg_counts"][h],
                        self.per_horizon["peg_counts"][h],
                        self.per_horizon["segz_counts"][h],
                        self.per_horizon["hypo_tp"][h],
                        self.per_horizon["hypo_fp"][h],
                        self.per_horizon["hypo_fn"][h],
                        self.per_horizon["hyper_tp"][h],
                        self.per_horizon["hyper_fp"][h],
                        self.per_horizon["hyper_fn"][h],
                        self.per_horizon["sev_hypo_tp"][h],
                        self.per_horizon["sev_hypo_fp"][h],
                        self.per_horizon["sev_hypo_fn"][h],
                        icgm_h,
                    )
                )
            result["per_horizon"] = ph
        if self.per_patient:
            result["per_patient"] = {
                pid: self._finalize_counts(
                    entry["count"],
                    entry["sum_sq"],
                    entry["sum_abs"],
                    entry["sum_seg"],
                    entry["sum_gse"],
                    entry["ceg_counts"],
                    entry["peg_counts"],
                    entry["segz_counts"],
                    entry.get("hypo_tp", 0.0),
                    entry.get("hypo_fp", 0.0),
                    entry.get("hypo_fn", 0.0),
                    entry.get("hyper_tp", 0.0),
                    entry.get("hyper_fp", 0.0),
                    entry.get("hyper_fn", 0.0),
                    entry.get("sev_hypo_tp", 0.0),
                    entry.get("sev_hypo_fp", 0.0),
                    entry.get("sev_hypo_fn", 0.0),
                    entry.get("icgm_counts", _init_icgm_counts()),
                )
                for pid, entry in self.per_patient.items()
            }
        return result
