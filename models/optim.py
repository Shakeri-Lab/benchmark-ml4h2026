from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def _as_float(v: Any, default: float) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off", "none", "null"}:
        return False
    return default


def build_optimizer(model: torch.nn.Module, optim_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    optim_cfg = dict(optim_cfg or {})
    name = str(optim_cfg.get("name", "adamw")).strip().lower()
    lr = _as_float(optim_cfg.get("lr", optim_cfg.get("learning_rate", 1e-3)), 1e-3)
    weight_decay = _as_float(optim_cfg.get("weight_decay", 0.0), 0.0)

    if name == "adam":
        betas = optim_cfg.get("betas", None)
        if betas is None:
            betas = (_as_float(optim_cfg.get("beta1", 0.9), 0.9), _as_float(optim_cfg.get("beta2", 0.999), 0.999))
        eps = _as_float(optim_cfg.get("eps", 1e-8), 1e-8)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas), eps=eps)

    if name == "adamw":
        betas = optim_cfg.get("betas", None)
        if betas is None:
            betas = (
                _as_float(optim_cfg.get("beta1", optim_cfg.get("adamw_beta1", 0.9)), 0.9),
                _as_float(optim_cfg.get("beta2", optim_cfg.get("adamw_beta2", 0.999)), 0.999),
            )
        eps = _as_float(optim_cfg.get("eps", optim_cfg.get("adamw_eps", 1e-8)), 1e-8)
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas), eps=eps)

    if name == "sgd":
        momentum = _as_float(optim_cfg.get("momentum", 0.0), 0.0)
        nesterov = _as_bool(optim_cfg.get("nesterov", False), False)
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

    raise ValueError(f"Unsupported optimizer '{name}'. Supported: adam, adamw, sgd")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: Optional[Dict[str, Any]],
    *,
    epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if not sched_cfg:
        return None
    sched_cfg = dict(sched_cfg or {})
    name = str(sched_cfg.get("name", "constant")).strip().lower()
    if name in {"none", "constant", ""}:
        return None

    base_lr = float(optimizer.param_groups[0]["lr"])
    min_lr = _as_float(sched_cfg.get("min_lr", sched_cfg.get("scheduler_min_lr", 0.0)), 0.0)

    if name == "linear":
        def lr_lambda(epoch: int) -> float:
            if epochs <= 1:
                return 1.0
            frac = max(0.0, 1.0 - (epoch / max(1, epochs - 1)))
            if base_lr <= 0:
                return frac
            return max(frac, min_lr / base_lr)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)), eta_min=min_lr)

    if name == "reduce_on_plateau":
        patience = _as_int(sched_cfg.get("patience", sched_cfg.get("scheduler_patience", 2)), 2)
        factor = _as_float(sched_cfg.get("factor", sched_cfg.get("scheduler_factor", 0.5)), 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor, min_lr=min_lr
        )

    raise ValueError(f"Unsupported scheduler '{name}'. Supported: constant, linear, cosine, reduce_on_plateau")


def build_optim_and_sched(
    model: torch.nn.Module,
    training_cfg: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    training_cfg = dict(training_cfg or {})
    optim_cfg = dict(training_cfg.get("optimizer") or {})
    sched_cfg = training_cfg.get("scheduler", None)
    epochs = _as_int(training_cfg.get("epochs", 1), 1)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, sched_cfg, epochs=epochs)
    return optimizer, scheduler

