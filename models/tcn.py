from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        d = torch.device(prefer)
        if d.type == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    logging.warning("GPU not available; using CPU.")
    return torch.device("cpu")


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv(x)
        # causal crop
        crop = out[:, :, :-self.conv.padding[0]] if self.conv.padding[0] > 0 else out
        out = self.relu(crop)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        res = res[..., -out.shape[-1] :]  # align length
        return out + res


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        history_len: int | None = None,
        pred_len: int | None = None,
        hidden_channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2**i
            layers.append(TemporalBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))
            in_ch = hidden_channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, 1)  # per-step prediction
        self.device = _get_device(device)
        self.to(self.device)

    def forward(self, x, pred_len: int):
        # x: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        feats = self.network(x)
        last = feats[:, :, -1]  # [B, hidden]
        step_pred = self.head(last)  # [B, 1]
        preds = step_pred.repeat(1, pred_len)  # [B, pred_len]
        return preds

    def predict(self, batch_X: np.ndarray, pred_len: int) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(batch_X, dtype=torch.float32, device=self.device)
            preds = self.forward(x, pred_len)
            return preds.cpu().numpy()
