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


class GRUPPPModel(nn.Module):
    """GRU baseline with RevIn normalization, DLinear-style decomposition, and patchified input."""

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
        patch_len: int = 16,
        patch_stride: int = 8,
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
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)

        if self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive")
        if self.history_len < self.patch_len:
            raise ValueError("history_len must be >= patch_len")

        self.patch_count = 1 + (self.history_len - self.patch_len) // self.patch_stride
        if self.patch_count <= 0:
            raise ValueError("Invalid patch_count computed from history_len/patch_len/patch_stride")

        in_dim = self.patch_len
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
        self.trend_head = nn.Linear(self.patch_count, self.pred_len)
        self.trend_head.weight = nn.Parameter(
            (1.0 / self.patch_count) * torch.ones([self.pred_len, self.patch_count])
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
        # Univariate-only for now; if multivariate, use the first channel.
        if x.ndim == 3 and x.shape[-1] > 1:
            x = x[..., :1]
        return x

    def _revin(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        std = std.clamp_min(self.revin_eps)
        x_norm = (x - mean) / std
        return x_norm, mean, std

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] -> [B, Np, patch_len]
        patches = x.unfold(1, self.patch_len, self.patch_stride)
        return patches

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        x = self._prepare_inputs(x, pred_len)
        x_norm, mean, std = self._revin(x)

        # Patchify normalized univariate series
        series = x_norm[..., 0] if x_norm.ndim == 3 else x_norm
        patches = self._patchify(series)
        if patches.shape[1] != self.patch_count:
            raise ValueError(
                f"Patch count mismatch: got {patches.shape[1]}, expected {self.patch_count}. "
                "Check history_len/patch_len/patch_stride."
            )

        seasonal, trend = self.decomp(patches)

        _, h_n = self.gru(seasonal)
        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, h_n.size(1), h_n.size(2))
            h_last = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)
        else:
            h_last = h_n[-1]

        if self.norm is not None:
            h_last = self.norm(h_last)

        seasonal_pred = self.head(h_last)

        trend_mean = trend.mean(dim=2)
        trend_pred = self.trend_head(trend_mean)

        pred_norm = seasonal_pred + trend_pred

        mean_cgm = mean[:, 0, 0].unsqueeze(-1)
        std_cgm = std[:, 0, 0].unsqueeze(-1)
        return pred_norm * std_cgm + mean_cgm
