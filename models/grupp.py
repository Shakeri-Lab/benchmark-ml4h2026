from __future__ import annotations

from typing import Any

import torch
from torch import nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block (seasonal + trend)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class GRUPPModel(nn.Module):
    """GRU baseline with RevIn normalization and DLinear-style decomposition."""

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
        decomposition_kernel_size: int = 25,
        revin_eps: float = 1.0e-6,
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
        self.decomposition_kernel_size = int(decomposition_kernel_size)
        self.revin_eps = float(revin_eps)

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

        self.decomp = series_decomp(self.decomposition_kernel_size)
        self.trend_head = nn.Linear(self.history_len, self.pred_len)
        self.trend_head.weight = nn.Parameter(
            (1.0 / self.history_len) * torch.ones([self.pred_len, self.history_len])
        )

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

    def _revin(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        std = std.clamp_min(self.revin_eps)
        x_norm = (x - mean) / std
        return x_norm, mean, std

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        x = self._prepare_inputs(x, pred_len)
        x_norm, mean, std = self._revin(x)

        seasonal, trend = self.decomp(x_norm)

        _, h_n = self.gru(seasonal)
        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, h_n.size(1), h_n.size(2))
            h_last = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)
        else:
            h_last = h_n[-1]

        if self.norm is not None:
            h_last = self.norm(h_last)

        seasonal_pred = self.head(h_last)

        trend_cgm = trend[..., 0]
        trend_pred = self.trend_head(trend_cgm)

        pred_norm = seasonal_pred + trend_pred

        mean_cgm = mean[:, 0, 0].unsqueeze(-1)
        std_cgm = std[:, 0, 0].unsqueeze(-1)
        return pred_norm * std_cgm + mean_cgm
