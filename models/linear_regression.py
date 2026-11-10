from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """
    Simple linear regression over flattened history*channels -> pred_len CGM.
    Trained via standard optimizer in the harness; can still be used with predict().
    """

    def __init__(self, input_dim: int, history_len: int, pred_len: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.history_len = history_len
        self.pred_len = pred_len
        self.linear = nn.Linear(input_dim * history_len, pred_len)

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        # x shape: [B, H, C]
        flat = x.reshape(x.shape[0], -1)
        return self.linear(flat)

    def predict(self, batch_X: np.ndarray, pred_len: int) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.as_tensor(batch_X, dtype=torch.float32, device=device)
            preds = self.forward(x, pred_len)
        return preds.cpu().numpy()
