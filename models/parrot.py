from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util

import numpy as np
import torch
from torch import nn


def _upstream_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "parroting_upstream"


def _load_upstream_parrot():
    repo_root = _upstream_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Vendored parroting repo not found at {repo_root}. "
            "Run the vendor step to populate third_party/parroting_upstream."
        )
    module_path = repo_root / "models" / "parrot.py"
    if not module_path.exists():
        raise FileNotFoundError(f"parrot.py not found at {module_path}")
    spec = importlib.util.spec_from_file_location("parroting_parrot_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load parrot.py from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


class ParrotModel(nn.Module):
    """Wrapper around the Parrot zero-shot forecaster."""

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        D: int = 30,
        tau: int = 1,
        context_len: int | None = None,
        univariate: bool | None = None,
        **_: Any,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.D = int(D)
        self.tau = int(tau)
        self.context_len = int(context_len) if context_len is not None else None
        self.univariate = univariate
        self.trainable = False

        mod = _load_upstream_parrot()
        self._context_parroting_forecast = mod.context_parroting_forecast

    def _prepare_inputs(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")
        if self.univariate:
            x = x[..., :1]
        return x

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        x = self._prepare_inputs(x, pred_len)
        ctx_len = self.context_len or x.shape[1]
        preds = []
        for b in range(x.shape[0]):
            context = x[b, -ctx_len:, :].detach().cpu().numpy()
            _, _, forecast = self._context_parroting_forecast(
                context[:, 0], D=self.D, tau=self.tau, forecast_total_length=self.pred_len
            )
            preds.append(np.asarray(forecast))
        out = torch.tensor(np.stack(preds, axis=0), dtype=torch.float32, device=x.device)
        return out
