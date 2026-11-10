from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import torch
import torch.nn as nn


class TimesFMModel(nn.Module):
    """Thin wrapper around TimesFM 2.5 (torch) for univariate CGM forecasting."""

    trainable = False

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        context_len: int | None = None,
        univariate: bool = False,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 1024,
        max_horizon: int = 256,
        normalize_inputs: bool = True,
        device: str = "auto",
        **_: Any,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.context_len = int(context_len) if context_len is not None else None
        self.univariate = bool(univariate)
        self.input_dim = int(input_dim)
        self._warned_multivariate = False
        self._warned_numpy_fallback = False

        try:
            import timesfm
        except Exception as e:
            raise ImportError(
                "Failed to import timesfm. Please install the official timesfm package."
            ) from e

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_id, device=self._device
        )
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=int(max_context),
                max_horizon=int(max_horizon),
                normalize_inputs=bool(normalize_inputs),
            )
        )

    def _prepare_inputs(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")

        # TimesFM is univariate. If multivariate input is provided, use CGM only.
        if x.ndim == 3 and x.shape[-1] > 1:
            if not self._warned_multivariate:
                warnings.warn(
                    "TimesFM is univariate; using the first (CGM) channel only.",
                    RuntimeWarning,
                )
                self._warned_multivariate = True
            x = x[..., 0]
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]

        if x.ndim != 2:
            raise ValueError(f"Unexpected input shape for TimesFM: {x.shape}")

        if self.context_len is not None and x.shape[1] > self.context_len:
            x = x[:, -self.context_len :]

        return x.to(self._device)

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        batch_input = self._prepare_inputs(x, pred_len)

        with torch.no_grad():
            try:
                point_forecast, _ = self._model.forecast(
                    horizon=self.pred_len,
                    inputs=batch_input,
                )
            except Exception:
                # Some versions only accept numpy; fallback with a warning once.
                if not self._warned_numpy_fallback:
                    warnings.warn(
                        "TimesFM forecast did not accept torch.Tensor inputs; falling back to numpy (CPU).",
                        RuntimeWarning,
                    )
                    self._warned_numpy_fallback = True
                point_forecast, _ = self._model.forecast(
                    horizon=self.pred_len,
                    inputs=batch_input.detach().cpu().numpy(),
                )

        if isinstance(point_forecast, torch.Tensor):
            preds = point_forecast
        else:
            preds = torch.from_numpy(np.asarray(point_forecast, dtype=np.float32))

        preds = preds.to(device=x.device, dtype=x.dtype)
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        return preds
