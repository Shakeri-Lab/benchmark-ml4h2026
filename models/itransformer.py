from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any, Tuple
import sys

import torch
from torch import nn


def _upstream_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "itransformer_upstream"


def _load_upstream_model_class():
    repo_root = _upstream_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Vendored iTransformer repo not found at {repo_root}. "
            "Run the vendor step to populate third_party/itransformer_upstream."
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from model.iTransformer import Model  # type: ignore

    return Model


def _build_upstream_config(
    *,
    seq_len: int,
    pred_len: int,
    d_model: int = 512,
    n_heads: int = 8,
    e_layers: int = 2,
    d_ff: int = 2048,
    dropout: float = 0.1,
    factor: int = 1,
    activation: str = "gelu",
    embed: str = "timeF",
    freq: str = "h",
    output_attention: bool = False,
    use_norm: bool = True,
    class_strategy: str = "projection",
) -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=int(seq_len),
        pred_len=int(pred_len),
        d_model=int(d_model),
        n_heads=int(n_heads),
        e_layers=int(e_layers),
        d_ff=int(d_ff),
        dropout=float(dropout),
        factor=int(factor),
        activation=str(activation),
        embed=str(embed),
        freq=str(freq),
        output_attention=bool(output_attention),
        use_norm=bool(use_norm),
        class_strategy=str(class_strategy),
    )


class ITransformerModel(nn.Module):
    """
    Thin wrapper around the official iTransformer implementation.
    """

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        univariate: bool | None = None,
        **params: Any,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.univariate = univariate

        cfg = _build_upstream_config(seq_len=self.history_len, pred_len=self.pred_len, **params)
        self.config = cfg
        Model = _load_upstream_model_class()
        self.model = Model(cfg)

    def _prepare_inputs(self, x: torch.Tensor, pred_len: int | None = None) -> Tuple[torch.Tensor, None, None, None]:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")
        if self.univariate:
            x = x[..., :1]
        return x, None, None, None

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        x_enc, x_mark_enc, x_dec, x_mark_dec = self._prepare_inputs(x, pred_len)
        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # out: [B, pred_len, N] -> return target channel [B, pred_len]
        return out[:, -self.pred_len :, 0]

