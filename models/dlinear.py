from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import importlib.util

import torch
from torch import nn


def _upstream_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "dlinear_upstream"


def _load_upstream_model_class():
    repo_root = _upstream_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Vendored DLinear repo not found at {repo_root}. "
            "Run the vendor step to populate third_party/dlinear_upstream."
        )
    module_path = repo_root / "models" / "DLinear.py"
    if not module_path.exists():
        raise FileNotFoundError(f"DLinear module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("dlinear_upstream_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DLinear module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.Model


class DLinearModel(nn.Module):
    """
    Thin wrapper around the official DLinear implementation.
    """

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        individual: bool = False,
        univariate: bool | None = None,
        **_: Any,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.individual = bool(individual)
        self.univariate = univariate

        enc_in = 1 if self.univariate else self.input_dim
        cfg = SimpleNamespace(
            seq_len=self.history_len,
            pred_len=self.pred_len,
            enc_in=int(enc_in),
            individual=self.individual,
        )
        self.config = cfg
        Model = _load_upstream_model_class()
        self.model = Model(cfg)

    def _prepare_inputs(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")
        if self.univariate:
            x = x[..., :1]
        return x

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        x = self._prepare_inputs(x, pred_len)
        out = self.model(x)
        if out.dim() != 3:
            raise ValueError(f"Unexpected DLinear output shape: {out.shape}")
        # Return CGM channel only to match target format [B, pred_len]
        return out[:, :, 0]
