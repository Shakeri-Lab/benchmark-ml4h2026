from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


def _require_statsmodels():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError(
            "statsmodels is required for ARIMAX. Please install statsmodels to use this model."
        ) from e
    return SARIMAX


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class ARIMAXModel(nn.Module):
    """ARIMAX baseline using statsmodels SARIMAX.

    Supports a global fit (default) or per-sample local fit inside forward().
    """

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        order: tuple[int, int, int] = (2, 1, 2),
        trend: Optional[str] = "c",
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = True,
        fit_strategy: str = "global",
        future_exog_strategy: str = "auto",
        drop_static_exog: bool = True,
        static_exog_tol: float = 1e-3,
        normalize_insulin_by_weight: bool = False,
        cf_shifted_exog_idx: Optional[Iterable[int]] = None,
        weight_idx: Optional[int] = None,
        insulin_idx: Optional[Iterable[int]] = None,
        rate_channel_nonzero_frac: float = 0.5,
        max_abs_coef: float = 1000.0,
        boundary_gap_len: int = 1,
        fit_maxiter: Optional[int] = None,
        fit_method: Optional[str] = None,
        fit_sample_limit: Optional[int] = None,
        fit_sample_seed: Optional[int] = 42,
        univariate: bool | None = None,
        **_: Any,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.order = tuple(order)
        self.trend = trend
        self.enforce_stationarity = bool(enforce_stationarity)
        self.enforce_invertibility = bool(enforce_invertibility)
        self.fit_strategy = str(fit_strategy)
        self.future_exog_strategy = str(future_exog_strategy)
        self.drop_static_exog = bool(drop_static_exog)
        self.static_exog_tol = float(static_exog_tol)
        self.normalize_insulin_by_weight = bool(normalize_insulin_by_weight)
        # Exog columns (indices into x[..., 1:]) that the CF loader left-shifted
        # by pred_len. Set for the interventional arm; unset elsewhere.
        self.cf_shifted_exog_idx = tuple(int(i) for i in cf_shifted_exog_idx) if cf_shifted_exog_idx else ()
        # Benchmark schema is [cgm, basal, bolus_standard, bolus_extended,
        # weight_kg, meal]; fall back to no normalisation on other layouts.
        if weight_idx is None and insulin_idx is None and self.input_dim == 6:
            weight_idx, insulin_idx = 4, (1, 2, 3)
        self.weight_idx = None if weight_idx is None else int(weight_idx)
        self.insulin_idx = tuple(int(i) for i in insulin_idx) if insulin_idx else ()
        self.rate_channel_nonzero_frac = float(rate_channel_nonzero_frac)
        self.max_abs_coef = float(max_abs_coef)
        self.boundary_gap_len = int(boundary_gap_len)
        self.fit_maxiter = fit_maxiter
        self.fit_method = fit_method
        self.fit_sample_limit = fit_sample_limit
        self.fit_sample_seed = fit_sample_seed
        self.univariate = univariate

        # Stored global params for global fit strategy
        self.register_buffer("global_params", torch.empty(0))
        # Which exog columns survive static-channel pruning, and which of those
        # are persisting rates (held over the horizon) rather than sparse events
        # (zeroed). Determined in prefit and carried in the checkpoint so that
        # evaluation reproduces the fitted design matrix exactly.
        self.register_buffer("exog_keep", torch.empty(0, dtype=torch.long))
        self.register_buffer("exog_is_rate", torch.empty(0, dtype=torch.bool))
        self._fitted = False
        self._exog_dim: Optional[int] = None
        # Ensure optimizer has a parameter even though ARIMAX is non-parametric
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def load_state_dict(self, state_dict, strict: bool = True):
        for name in ("exog_keep", "exog_is_rate"):
            if name in state_dict:
                buf = state_dict[name]
                cur = getattr(self, name, None)
                if cur is None or cur.shape != buf.shape:
                    setattr(self, name, buf.detach().clone())
        if "global_params" in state_dict:
            gp = state_dict["global_params"]
            if getattr(self, "global_params", None) is None or self.global_params.shape != gp.shape:
                self.global_params = gp.detach().clone()
        out = super().load_state_dict(state_dict, strict=strict)
        self._fitted = bool(getattr(self, "global_params", torch.empty(0)).numel() > 0)
        return out

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.univariate:
            return x[..., :1]
        if self.normalize_insulin_by_weight and self.weight_idx is not None and self.insulin_idx:
            if x.shape[-1] > max(self.weight_idx, *self.insulin_idx):
                x = x.clone()
                w = x[..., self.weight_idx : self.weight_idx + 1]
                # Guard against missing/zero weights; those windows keep raw units.
                w = torch.where(w > 1.0, w, torch.ones_like(w))
                for i in self.insulin_idx:
                    x[..., i : i + 1] = x[..., i : i + 1] / w
        return x

    def _select_exog(self, exog: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Restrict to the exog columns retained at fit time."""
        if exog is None:
            return None
        if self.exog_keep.numel() == 0:
            return exog
        idx = self.exog_keep.cpu().numpy()
        # A checkpoint's exog_keep is sized to the exog width it was FITTED with.
        # Silently dropping out-of-range indices misaligns the saved coefficient
        # vector against the supplied regressors: the AR terms are then read from
        # the wrong positions and the forecast diverges (observed: RMSE ~1e12 at
        # 48 steps) with no error. Fail loudly instead.
        if idx.size and int(idx.max()) >= exog.shape[1]:
            raise ValueError(
                "ARIMAX exog width mismatch: checkpoint was fitted with at least "
                f"{int(idx.max()) + 1} exogenous channels (exog_keep={idx.tolist()}) "
                f"but {exog.shape[1]} were supplied. The checkpoint is invalid for "
                "this feature set -- refit the model."
            )
        if idx.size == 0:
            return None
        return exog[:, idx]

    def _build_exog_future(
        self,
        exog_hist: Optional[np.ndarray],
        provided: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if exog_hist is None:
            return None
        k = exog_hist.shape[1]
        strategy = self.future_exog_strategy

        if strategy == "provided":
            if provided is None:
                raise ValueError(
                    "future_exog_strategy='provided' requires future_exog to be passed to forward()."
                )
            fut = np.asarray(provided, dtype=float)
            if fut.ndim == 1:
                fut = fut[:, None]
            if fut.shape[1] != k:
                fut = self._select_exog(fut)
            if fut is None or fut.shape != (self.pred_len, k):
                raise ValueError(
                    f"provided future exog has shape {None if fut is None else fut.shape}, expected {(self.pred_len, k)}"
                )
            return fut.astype(exog_hist.dtype)

        if strategy == "zeros":
            return np.zeros((self.pred_len, k), dtype=exog_hist.dtype)
        if strategy == "repeat_last":
            return np.repeat(exog_hist[-1:, :], self.pred_len, axis=0)
        if strategy == "auto":
            # Persisting rates (e.g. basal) are held at their last value; sparse
            # event channels (bolus, meals) have no known future occurrence and
            # are zero. Applying one rule to both is wrong in one direction or
            # the other.
            out = np.zeros((self.pred_len, k), dtype=exog_hist.dtype)
            if self.exog_is_rate.numel() == k:
                rate = self.exog_is_rate.cpu().numpy()
            else:
                rate = np.zeros(k, dtype=bool)
            if rate.any():
                out[:, rate] = np.repeat(exog_hist[-1:, rate], self.pred_len, axis=0)
            return out
        raise ValueError(f"Unknown future_exog_strategy '{self.future_exog_strategy}'")

    def _configured_trend(self) -> Optional[str]:
        if self.trend is None:
            return None
        trend = str(self.trend).strip().lower()
        if trend in {"none", "null"}:
            return None
        return trend

    @staticmethod
    def _as_2d_exog(exog: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if exog is None:
            return None
        arr = np.asarray(exog, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D exog array, got shape {arr.shape}")
        return arr

    def _use_explicit_const_exog(self, exog: Optional[np.ndarray]) -> bool:
        arr = self._as_2d_exog(exog)
        return arr is not None and self._configured_trend() == "c"

    def _effective_trend(self, exog: Optional[np.ndarray]) -> Optional[str]:
        trend = self._configured_trend()
        if trend is None:
            return None
        if self._use_explicit_const_exog(exog):
            return "n"
        return trend

    def _format_exog(self, exog: Optional[np.ndarray]) -> Optional[pd.DataFrame]:
        arr = self._as_2d_exog(exog)
        if arr is None:
            return None
        data = arr
        columns = [f"exog_{i}" for i in range(arr.shape[1])]
        if self._use_explicit_const_exog(arr):
            data = np.concatenate([np.ones((arr.shape[0], 1), dtype=arr.dtype), arr], axis=1)
            columns = ["trend_const", *columns]
        return pd.DataFrame(data, columns=columns)

    def prefit(self, train_loader: Iterable, val_loader: Optional[Iterable] = None):
        if self.fit_strategy != "global":
            self._fitted = True
            return

        SARIMAX = _require_statsmodels()
        endog_chunks = []
        exog_chunks = []
        gap = max(self.boundary_gap_len, 0)

        sample_limit = None if self.fit_sample_limit in (None, -1, "null") else int(self.fit_sample_limit)
        rng = np.random.default_rng(self.fit_sample_seed if self.fit_sample_seed is not None else 42)
        reservoir: list[tuple[np.ndarray, Optional[np.ndarray]]] = []
        seen = 0

        for batch in tqdm(train_loader, desc="ARIMAX prefit", leave=False):
            X = batch[0]
            X = self._prepare_inputs(X)
            x_np = _as_numpy(X)
            for b in range(x_np.shape[0]):
                endog = x_np[b, :, 0]
                exog = x_np[b, :, 1:] if x_np.shape[-1] > 1 else None
                if sample_limit is None:
                    reservoir.append((endog, exog))
                    continue
                seen += 1
                if len(reservoir) < sample_limit:
                    reservoir.append((endog, exog))
                else:
                    j = int(rng.integers(0, seen))
                    if j < sample_limit:
                        reservoir[j] = (endog, exog)

        self._analyse_exog_channels(reservoir)
        reservoir = [(e, self._select_exog(x)) for e, x in reservoir]

        for endog, exog in reservoir:
            endog_chunks.append(endog)
            if exog is not None:
                exog_chunks.append(exog)
            if gap > 0:
                endog_chunks.append(np.full(gap, np.nan))
                if exog is not None:
                    exog_chunks.append(np.zeros((gap, exog.shape[1])))

        endog_all = np.concatenate(endog_chunks, axis=0) if endog_chunks else np.array([])
        if endog_all.size == 0:
            raise ValueError("No data available to fit ARIMAX global parameters.")

        exog_all = None
        if exog_chunks:
            exog_all = np.concatenate(exog_chunks, axis=0)
            self._exog_dim = exog_all.shape[1]
        else:
            self._exog_dim = None
        exog_all_fmt = self._format_exog(exog_all)

        mod = SARIMAX(
            endog_all,
            exog=exog_all_fmt,
            order=self.order,
            trend=self._effective_trend(exog_all),
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            missing="drop",
        )
        fit_kwargs: dict[str, Any] = {"disp": False}
        if self.fit_maxiter is not None:
            fit_kwargs["maxiter"] = int(self.fit_maxiter)
        if self.fit_method is not None:
            fit_kwargs["method"] = str(self.fit_method)

        pbar = tqdm(total=fit_kwargs.get("maxiter", None), desc="ARIMAX fit", leave=False)

        def _cb(_params):
            pbar.update(1)

        fit_kwargs["callback"] = _cb
        res = mod.fit(**fit_kwargs)
        pbar.close()
        self._check_fit(res)
        params = np.asarray(res.params)
        self.global_params = torch.tensor(params, dtype=torch.float64)
        self._fitted = True

    def _analyse_exog_channels(self, reservoir) -> None:
        """Decide which exog columns to keep and which behave as rates.

        A channel that is constant within every window carries no within-window
        information and is collinear with the constant term that _format_exog
        adds under trend="c". Keeping it makes the design singular and, worse,
        any future-exog rule that changes its value injects a step at the
        forecast origin proportional to its coefficient.
        """
        cols = [x for _, x in reservoir if x is not None]
        if not cols:
            self.exog_keep = torch.empty(0, dtype=torch.long)
            self.exog_is_rate = torch.empty(0, dtype=torch.bool)
            return
        k = cols[0].shape[1]
        within_cv = np.zeros(k)
        nonzero_frac = np.zeros(k)
        for arr in cols:
            mu = np.abs(arr.mean(axis=0))
            within_cv += arr.std(axis=0) / np.maximum(mu, 1e-9)
            nonzero_frac += (np.abs(arr) > 0).mean(axis=0)
        within_cv /= len(cols)
        nonzero_frac /= len(cols)
        within_std = within_cv  # reported below

        # Relative, not absolute: weight_kg is interpolated between measurement
        # dates so its within-window std is ~1e-3 against a mean of ~73.
        keep = [i for i in range(k)
                if not (self.drop_static_exog and within_cv[i] <= self.static_exog_tol)]
        dropped = [i for i in range(k) if i not in keep]
        if dropped:
            print(f"[ARIMAX] dropping static exog columns {dropped} "
                  f"(mean within-window std {[round(float(within_std[i]), 8) for i in dropped]})")
        self.exog_keep = torch.tensor(keep, dtype=torch.long)
        self.exog_is_rate = torch.tensor(
            [bool(nonzero_frac[i] >= self.rate_channel_nonzero_frac) for i in keep],
            dtype=torch.bool,
        )
        print(f"[ARIMAX] kept exog columns {keep}; "
              f"rate-like={[bool(b) for b in self.exog_is_rate.tolist()]} "
              f"(nonzero frac {[round(float(nonzero_frac[i]), 3) for i in keep]})")

    def _check_fit(self, res) -> None:
        """Reject a diverged optimum instead of filtering with it.

        The magnitude bound applies to the regression and ARMA coefficients only.
        The final SARIMAX parameter is the innovation variance, which for glucose
        in mg/dL^2 is legitimately in the tens to hundreds; including it rejected
        healthy fits. AR/MA validity is separately guaranteed by
        enforce_stationarity / enforce_invertibility.
        """
        params = np.asarray(res.params, dtype=float)
        if not np.all(np.isfinite(params)):
            raise RuntimeError(f"ARIMAX fit produced non-finite parameters: {params.tolist()}")
        coefs = params[:-1] if params.size > 1 else params  # drop sigma2
        bad = np.abs(coefs) > self.max_abs_coef
        converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
        if not converged:
            raise RuntimeError("ARIMAX global fit did not converge.")
        if bad.any():
            raise RuntimeError(
                f"ARIMAX fit produced implausible coefficients "
                f"(|coef| > {self.max_abs_coef}, excluding sigma2): {coefs[bad].round(3).tolist()}. "
                "This indicates a diverged fit."
            )

    def _forecast_single(self, endog: np.ndarray, exog: Optional[np.ndarray],
                         future_exog: Optional[np.ndarray] = None) -> np.ndarray:
        SARIMAX = _require_statsmodels()
        exog = self._select_exog(exog)
        exog_fmt = self._format_exog(exog)
        mod = SARIMAX(
            endog,
            exog=exog_fmt,
            order=self.order,
            trend=self._effective_trend(exog),
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            missing="drop",
        )

        if self.fit_strategy == "global":
            if self.global_params.numel() == 0:
                raise RuntimeError("ARIMAX global params not fitted. Call prefit() first or load a checkpoint.")
            self._fitted = True
            res = mod.filter(self.global_params.cpu().numpy())
        elif self.fit_strategy == "local":
            res = mod.fit(disp=False)
        else:
            raise ValueError(f"Unknown fit_strategy '{self.fit_strategy}'")

        exog_future = self._format_exog(self._build_exog_future(exog, provided=future_exog))
        forecast = res.get_forecast(steps=self.pred_len, exog=exog_future).predicted_mean
        return np.asarray(forecast)

    def _unshift_cf_window(self, endog: np.ndarray, exog: Optional[np.ndarray]):
        """Undo the CF left-shift, returning (endog, exog_hist, exog_future)."""
        L = self.pred_len
        H = endog.shape[0]
        if exog is None or L <= 0 or H <= L:
            return endog, exog, None
        k = exog.shape[1]
        shifted = set(self.cf_shifted_exog_idx)
        endog_al = endog[L:H]                       # cgm over times [L, H)
        hist = np.zeros((H - L, k), dtype=exog.dtype)
        fut = np.zeros((L, k), dtype=exog.dtype)
        for c in range(k):
            if c in shifted:
                hist[:, c] = exog[0:H - L, c]       # insulin over [L, H)
                fut[:, c] = exog[H - L:H, c]        # insulin over [H, H+L) -- the plan
            else:
                hist[:, c] = exog[L:H, c]           # already aligned
                # no known future for unshifted event channels
                if self.exog_is_rate.numel() == k and bool(self.exog_is_rate[c]):
                    fut[:, c] = exog[H - 1, c]
        return endog_al, hist, fut

    def forward(self, x: torch.Tensor, pred_len: int | None = None,
                future_exog: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")

        x = self._prepare_inputs(x)
        x_np = _as_numpy(x)
        preds = []
        for b in range(x_np.shape[0]):
            endog = x_np[b, :, 0]
            exog = None
            if x_np.shape[-1] > 1:
                exog = x_np[b, :, 1:]
            fut = None
            if future_exog is not None:
                fe = future_exog[b]
                fut = fe.detach().cpu().numpy() if hasattr(fe, "detach") else np.asarray(fe)
            elif self.future_exog_strategy == "provided" and self.cf_shifted_exog_idx:
                endog, exog, fut = self._unshift_cf_window(endog, exog)
            pred = self._forecast_single(endog, exog, future_exog=fut)
            preds.append(pred)
        out = torch.tensor(np.stack(preds, axis=0), dtype=torch.float32, device=x.device)
        return out
