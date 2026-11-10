from __future__ import annotations

import numpy as np


class ZeroOrderHoldModel:
    """Baseline: repeat the last observed target over the prediction horizon."""

    def __init__(self, **kwargs):
        pass

    def predict(self, batch_X: np.ndarray, pred_len: int) -> np.ndarray:
        last = batch_X[:, -1, 0]
        return np.repeat(last[:, None], pred_len, axis=1)

