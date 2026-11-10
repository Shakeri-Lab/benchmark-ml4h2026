from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import importlib.util
import types
import sys

import torch
from torch import nn


def _upstream_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "tfb_upstream"


def _load_upstream_model_class():
    repo_root = _upstream_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Vendored TFB repo not found at {repo_root}. "
            "Run the vendor step to populate third_party/tfb_upstream."
        )
    module_path = repo_root / "ts_benchmark" / "baselines" / "patchmlp" / "models" / "patchmlp_model.py"
    if not module_path.exists():
        raise FileNotFoundError(f"PatchMLP module not found at {module_path}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Avoid importing ts_benchmark.baselines.patchmlp/__init__.py (which pulls other baselines)
    if "ts_benchmark.baselines.patchmlp" not in sys.modules:
        pkg = types.ModuleType("ts_benchmark.baselines.patchmlp")
        pkg.__path__ = [str(repo_root / "ts_benchmark" / "baselines" / "patchmlp")]
        sys.modules["ts_benchmark.baselines.patchmlp"] = pkg
    if "ts_benchmark.baselines.patchmlp.models" not in sys.modules:
        pkg_models = types.ModuleType("ts_benchmark.baselines.patchmlp.models")
        pkg_models.__path__ = [str(repo_root / "ts_benchmark" / "baselines" / "patchmlp" / "models")]
        sys.modules["ts_benchmark.baselines.patchmlp.models"] = pkg_models

    spec = importlib.util.spec_from_file_location("ts_benchmark.baselines.patchmlp.models.patchmlp_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PatchMLP module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.PatchMLPModel


class PatchMLPModel(nn.Module):
    """
    Thin wrapper around the official PatchMLP implementation.
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
        **_: Any,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.univariate = univariate

        enc_in = 1 if self.univariate else self.input_dim
        cfg = SimpleNamespace(
            seq_len=self.history_len,
            pred_len=self.pred_len,
            enc_in=int(enc_in),
            d_model=int(d_model),
            e_layers=int(e_layers),
            use_norm=bool(use_norm),
            output_attention=bool(output_attention),
        )
        self.config = cfg
        Model = _load_upstream_model_class()
        self.model = Model(cfg)

    def _prepare_inputs(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")
        if self.univariate:
            x = x[..., :1]
        return x

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        x_enc = self._prepare_inputs(x, pred_len)
        out = self.model(x_enc, None, None, None)
        if out.dim() != 3:
            raise ValueError(f"Unexpected PatchMLP output shape: {out.shape}")
        return out[:, :, 0]
