from __future__ import annotations

import importlib
from typing import Any, Callable, Type

from .linear_regression import LinearRegressionModel
from .gru import GRUModel
from .tcn import TCNModel
from .zoh import ZeroOrderHoldModel


_MODEL_REGISTRY: dict[str, Any] = {
    "zoh": ZeroOrderHoldModel,
    "tcn": TCNModel,
    "linear_regression": LinearRegressionModel,
    "gru": GRUModel,
    "grupp": "models.grupp:GRUPPModel",
    "gruppp": "models.gruppp:GRUPPPModel",
    "itransformer": "models.itransformer:ITransformerModel",
    "garnn_gatv2_exp": "models.garnn_gatv2_exp:GARNNGATv2ExpModel",
    "dlinear": "models.dlinear:DLinearModel",
    "patchmlp": "models.patchmlp:PatchMLPModel",
    "patchmlp_glimmer": "models.patchmlp_glimmer:PatchMLPGlimmerModel",
    "attengluco": "models.attengluco:AttenGlucoModel",
    "arimax": "models.arimax.model:ARIMAXModel",
    "parrot": "models.parrot:ParrotModel",
    "timesfm": "models.timesfm:TimesFMModel",
    "glimmer": "models.glimmer:GlimmerModel",
}


def _resolve_model(entry: Any) -> Type:
    if isinstance(entry, str):
        module_path, cls_name = entry.split(":")
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)
    return entry


def get_model(name: str, **kwargs):
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    cls = _resolve_model(_MODEL_REGISTRY[name])
    return cls(**kwargs)


def build_model_from_cfg(model_cfg: dict, *, input_dim: int, history_len: int, pred_len: int):
    """
    Build a model from a config dict.

    Expected formats:
      model_cfg = {"name": "tcn", "params": {...}}
      or legacy: {"model": "tcn", ...}
    """
    model_cfg = dict(model_cfg or {})
    name = model_cfg.get("name") or model_cfg.get("model") or model_cfg.get("model_name")
    if not name:
        raise ValueError("model_cfg must include 'name'")
    params = model_cfg.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("model_cfg.params must be a dict")
    return get_model(str(name), input_dim=int(input_dim), history_len=int(history_len), pred_len=int(pred_len), **params)


def list_models():
    return list(_MODEL_REGISTRY.keys())
