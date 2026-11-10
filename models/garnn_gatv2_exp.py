from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import torch
from torch import nn


def _upstream_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "garnn_public_upstream"


def _load_upstream_model_class():
    repo_root = _upstream_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Vendored GARNN repo not found at {repo_root}. "
            "Run the vendor step to populate third_party/garnn_public_upstream."
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from gat_mem_org_log.method import MLP_GATV2_exp_MLP  # type: ignore

    return MLP_GATV2_exp_MLP


class GARNNGATv2ExpModel(nn.Module):
    """
    Thin wrapper around the official GARNN (gatv2_exp) implementation.
    """

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        univariate: bool | None = None,
        hidden_dim: int = 512,
        nhead: int = 1,
        num_layers: int = 2,
        all_edges: bool = True,
        alpha: float = 0.2,
        **_: Any,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.univariate = univariate
        self.num_varis = 1 if self.univariate else self.input_dim
        if self.num_varis <= 0:
            raise ValueError(f"Invalid num_varis={self.num_varis}")
        if (hidden_dim // self.num_varis) <= 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be >= num_varis ({self.num_varis})"
            )

        self.upstream_kwargs = {
            "hidden_dim": int(hidden_dim),
            "nhead": int(nhead),
            "num_varis": int(self.num_varis),
            "num_layers": int(num_layers),
            "all_edges": bool(all_edges),
            "alpha": float(alpha),
        }
        Model = _load_upstream_model_class()
        self.model = Model(**self.upstream_kwargs)

    def _prepare_inputs(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")
        if self.univariate:
            x = x[..., :1]
        if x.shape[-1] != self.num_varis:
            raise ValueError(f"Input channel mismatch: got {x.shape[-1]}, expected {self.num_varis}")
        return x

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")
        G = self._prepare_inputs(x, pred_len=None)
        device_type = "cuda" if G.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            out = self.model(G.float(), None, None, None)
        pred = out["pred"] if isinstance(out, dict) else out
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        elif pred.dim() > 2:
            pred = pred.reshape(pred.shape[0], -1)
        return pred
