from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import importlib.util

import torch
from torch import nn


def _upstream_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "attengluco_upstream"


def _load_upstream_model_class():
    repo_root = _upstream_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Vendored AttenGluco repo not found at {repo_root}. "
            "Run the vendor step to populate third_party/attengluco_upstream."
        )
    module_path = repo_root / "vendor_extracted" / "attengluco_model.py"
    if not module_path.exists():
        raise FileNotFoundError(f"AttenGluco module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("attengluco_upstream_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AttenGluco module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.TimeSeriesTransformer


def _as_idx_list(v: Any) -> list[int]:
    if v is None:
        return []
    if isinstance(v, int):
        return [int(v)]
    if isinstance(v, Sequence):
        return [int(i) for i in v]
    raise TypeError(f"Expected int or sequence of ints, got {type(v)!r}")


class AttenGlucoModel(nn.Module):
    """
    Wrapper around the official AttenGluco implementation.

    Upstream consumes exactly three streams: a primary channel A that
    cross-attends to two auxiliary channels B and C (in the AI-READI paper,
    glucose attending to walking steps and walking intervals). Running it on this
    benchmark requires fixing two things.

    Channel mapping
    ---------------
    The benchmark schema is
    ``[cgm, basal, bolus_standard, bolus_extended, weight_kg, meal]``. The
    previous mapping used B = basal and C = the *sum of all remaining raw
    channels*, which placed the static covariate ``weight_kg`` (order 10-100)
    into the same sum as sparse ``bolus``/``meal`` events (order 0-10). C became
    an essentially constant per-patient scalar and the insulin signal was lost.

    Basal and bolus are kept as separate streams: although a pump renders both
    from the same insulin preparation, they play different control-theoretic
    roles, and upstream's two auxiliary slots accommodate both without
    collapsing them. The mapping is

        A = cgm
        B = basal                                   [U/hr]
        C = bolus_standard + bolus_extended         [U/step]

    ``bolus_standard`` and ``bolus_extended`` are two delivery profiles of the
    same bolus action and are summed. ``weight_kg`` is dropped as a static
    covariate. Because each stream is standardized independently (see below),
    the U/hr vs U/step unit difference between B and C is immaterial.

    The casualty is the carbohydrate channel: the reference three-stream
    interface has no fourth slot, so AttenGluco cannot ingest meals without
    modifying the architecture. This is a genuine limitation of the published
    design; it is inert on DCLP3/DCLP5 (where ``meal`` is identically zero) but
    does withhold real signal on PEDAP. Set ``collapse_insulin=True`` to instead
    merge basal and bolus into B and route meals to C.

    Normalization
    -------------
    The upstream architecture terminates in a LayerNorm and regresses the horizon
    from a flattened normalized representation, so it assumes scaled inputs and
    targets: the reference training script MinMax-scales glucose to [0, 1] and
    inverse-transforms its predictions. Fed raw mg/dL it cannot restore the
    absolute glucose level, which is the dominant predictor at short horizons.

    ``use_norm`` applies per-window, per-stream instance normalization and
    de-normalizes the output with the target stream's statistics, matching the
    ``use_norm=True`` convention already used by the iTransformer and PatchMLP
    wrappers in this repo, so all deep baselines see inputs on the same scale.
    """

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        d_model: int = 32,
        n_heads: int = 4,
        ff_units: int = 128,
        univariate: bool | None = None,
        use_norm: bool = True,
        norm_eps: float = 1e-5,
        cgm_idx: int = 0,
        basal_idx: Any = 1,
        bolus_idx: Any = (2, 3),
        meal_idx: Any = (5,),
        collapse_insulin: bool = False,
        basal_rate_divisor: float = 12.0,
        **_: Any,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.univariate = univariate
        self.use_norm = bool(use_norm)
        self.norm_eps = float(norm_eps)
        self.n_features = 3

        self.cgm_idx = int(cgm_idx)
        # Keep only indices the incoming tensor actually has, so the wrapper also
        # works on cohorts with a reduced channel set.
        self.basal_idx = [i for i in _as_idx_list(basal_idx) if i < self.input_dim]
        self.bolus_idx = [i for i in _as_idx_list(bolus_idx) if i < self.input_dim]
        self.meal_idx = [i for i in _as_idx_list(meal_idx) if i < self.input_dim]
        self.collapse_insulin = bool(collapse_insulin)
        self.basal_rate_divisor = float(basal_rate_divisor)

        cfg = SimpleNamespace(
            n_time_steps=self.history_len,
            n_features=self.n_features,
            d_model=int(d_model),
            n_heads=int(n_heads),
            ff_units=int(ff_units),
            prediction_horizon=self.pred_len,
        )
        self.config = cfg
        Model = _load_upstream_model_class()
        self.model = Model(
            n_time_steps=cfg.n_time_steps,
            n_features=cfg.n_features,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ff_units=cfg.ff_units,
            prediction_horizon=cfg.prediction_horizon,
        )

    def _sum_channels(self, x: torch.Tensor, idx: list[int], like: torch.Tensor,
                      divisor: float = 1.0) -> torch.Tensor:
        out = torch.zeros_like(like)
        for i in idx:
            out = out + x[..., i : i + 1] / divisor
        return out

    def _prepare_inputs(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        """Build the three upstream streams."""
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, T, C], got {x.shape}")

        a = x[..., self.cgm_idx : self.cgm_idx + 1]

        if self.univariate:
            zeros = torch.zeros_like(a)
            return torch.cat([a, zeros, zeros], dim=-1)

        if self.collapse_insulin:
            # B = total insulin per step (basal rate converted to U/step), C = meals.
            b = self._sum_channels(x, self.basal_idx, a, divisor=self.basal_rate_divisor)
            b = b + self._sum_channels(x, self.bolus_idx, a)
            c = self._sum_channels(x, self.meal_idx, a)
        else:
            # Default: keep the two insulin modes in separate streams.
            b = self._sum_channels(x, self.basal_idx, a)
            c = self._sum_channels(x, self.bolus_idx, a)

        return torch.cat([a, b, c], dim=-1)

    def _instance_norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-window, per-stream standardization over the time axis.

        Streams that are constant within a window (e.g. an all-zero meal channel
        on DCLP3/DCLP5) have zero variance and normalize to zero.
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_norm = (x - mean) / (std + self.norm_eps)
        # Stream 0 is the target (CGM); keep its stats for de-normalization.
        return x_norm, mean[:, :, 0], std[:, :, 0]

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        x = self._prepare_inputs(x, pred_len)

        mean = std = None
        if self.use_norm:
            x, mean, std = self._instance_norm(x)

        out = self.model(x)
        if out.dim() != 2:
            raise ValueError(f"Unexpected AttenGluco output shape: {out.shape}")

        if self.use_norm:
            out = out * (std + self.norm_eps) + mean
        return out
