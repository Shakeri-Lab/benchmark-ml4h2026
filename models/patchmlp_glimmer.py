from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .patchmlp import PatchMLPModel


class PatchMLPGlimmerModel(nn.Module):
    """
    PatchMLP backbone with GLIMMER-style glucose-weighted training.

    This wrapper keeps the PatchMLP architecture unchanged and relies on the
    training loop to apply the GLIMMER-style region-weighted loss. The model
    forward simply returns PatchMLP predictions (univariate CGM forecast).
    """

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        d_model: int = 1024,
        e_layers: int = 1,
        use_norm: bool = True,
        output_attention: bool = False,
        univariate: bool | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.backbone = PatchMLPModel(
            input_dim=input_dim,
            history_len=history_len,
            pred_len=pred_len,
            d_model=d_model,
            e_layers=e_layers,
            use_norm=use_norm,
            output_attention=output_attention,
            univariate=univariate,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        return self.backbone(x, pred_len)
