from __future__ import annotations

import torch


def mse_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((preds - target) ** 2)


def mae_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(preds - target))


def gmse_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Torch implementation of gMSE per-point penalty:
    gSE(g,ĝ) = (g - ĝ)^2 * Pen(g,ĝ), Pen as defined in metrics.clinical (translated to torch).
    """
    # Parameters from Table I
    alpha_l, alpha_h = 1.5, 1.0
    beta_l, beta_h = 30.0, 100.0
    gamma_l, gamma_h = 10.0, 20.0
    t_l, t_h = 85.0, 155.0

    g = target
    ghat = preds

    def sigma_ge(x, a, eps):
        xi = (2.0 / eps) * (x - a) - 1.0
        res = torch.where(
            x <= a,
            torch.zeros_like(x),
            torch.where(
                x <= a + eps / 2,
                -0.5 * xi**4 - xi**3 + xi + 0.5,
                torch.where(
                    x <= a + eps,
                    0.5 * xi**4 - xi**3 + xi + 0.5,
                    torch.ones_like(x),
                ),
            ),
        )
        return res

    def sigma_le_bar(x, a, eps):
        xi_bar = (2.0 / eps) * (x - a) + 1.0
        res = torch.where(
            x <= a - eps,
            torch.ones_like(x),
            torch.where(
                x <= a - eps / 2,
                0.5 * xi_bar**4 - xi_bar**3 + xi_bar + 0.5,
                torch.where(
                    x <= a,
                    -0.5 * xi_bar**4 - xi_bar**3 + xi_bar + 0.5,
                    torch.zeros_like(x),
                ),
            ),
        )
        return res

    pen = (
        1.0
        + alpha_l * sigma_le_bar(g, t_l, beta_l) * sigma_ge(ghat, g, gamma_l)
        + alpha_h * sigma_ge(g, t_h, beta_h) * sigma_le_bar(ghat, g, gamma_h)
    )
    gse = (g - ghat) ** 2 * pen
    return torch.mean(gse)


LOSS_REGISTRY = {
    "mse": mse_loss,
    "mae": mae_loss,
    "gmse": gmse_loss,
}


def glimmer_weighted_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    w_hypo: float = 2.0,
    w_hyper: float = 2.0,
    hypo_threshold: float = 70.0,
    hyper_threshold: float = 180.0,
    mode: str = "mse",
) -> torch.Tensor:
    """
    GLIMMER-style region-weighted MSE:
      - euglycemia weight fixed at 1.0
      - hypo: target < hypo_threshold
      - hyper: target > hyper_threshold
    """
    weights = torch.ones_like(target)
    weights = torch.where(target < hypo_threshold, torch.as_tensor(w_hypo, device=target.device), weights)
    weights = torch.where(target > hyper_threshold, torch.as_tensor(w_hyper, device=target.device), weights)
    err = preds - target
    if str(mode).lower() == "mae":
        base = torch.abs(err)
    else:
        base = err ** 2
    return torch.mean(weights * base)


def get_loss(name: str, **kwargs):
    key = str(name).lower()
    if key == "glimmer_weighted":
        return lambda preds, target: glimmer_weighted_mse(preds, target, **kwargs)
    if key not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys()) + ['glimmer_weighted']}")
    return LOSS_REGISTRY[key]
