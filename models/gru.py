from __future__ import annotations

from typing import Any

import torch
from torch import nn


class GRUModel(nn.Module):
    """Simple GRU baseline for CGM forecasting."""

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        use_layernorm: bool = False,
        head_type: str = "linear",
        head_hidden_dim: int = 128,
        head_num_layers: int = 2,
        univariate: bool | None = None,
        **_: Any,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.use_layernorm = bool(use_layernorm)
        self.head_type = str(head_type)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_num_layers = int(head_num_layers)
        self.univariate = univariate

        in_dim = 1 if self.univariate else self.input_dim
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim) if self.use_layernorm else None

        self.head = self._build_head(out_dim)

    def _build_head(self, in_dim: int) -> nn.Module:
        head_type = self.head_type.lower()
        if head_type == "linear":
            return nn.Linear(in_dim, self.pred_len)
        if head_type != "mlp":
            raise ValueError(f"Unsupported head_type '{self.head_type}'")
        layers = []
        dim = in_dim
        for _ in range(max(self.head_num_layers - 1, 0)):
            layers.append(nn.Linear(dim, self.head_hidden_dim))
            layers.append(nn.ReLU())
            dim = self.head_hidden_dim
        layers.append(nn.Linear(dim, self.pred_len))
        return nn.Sequential(*layers)

    def _prepare_inputs(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")
        if self.univariate:
            x = x[..., :1]
        return x

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        x = self._prepare_inputs(x, pred_len)
        _, h_n = self.gru(x)
        if self.bidirectional:
            # h_n: [num_layers*2, B, hidden]
            h_n = h_n.view(self.num_layers, 2, h_n.size(1), h_n.size(2))
            h_last = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)
        else:
            h_last = h_n[-1]

        if self.norm is not None:
            h_last = self.norm(h_last)

        return self.head(h_last)
