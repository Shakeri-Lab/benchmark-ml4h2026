import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal
from typing import Sequence
from src.models.em import EMModel
from src.utils.device import clear_device

class MAR(EMModel):
    def __init__(
        self, 
        n_features: int,
        p: int, 
        k: int,
        random_seed: int, 
        init_sigma: float | None = None,
        alpha: float = 3.0,
        beta: float = 2.0,
        lam: Sequence[float] | np.ndarray = [1e-1],
        dir_alpha: float = 5.0,
        *args, 
        **kwargs
    ) -> None:
        component_kwargs = {
            "n_features": n_features,
            "p": p,
            "init_sigma": init_sigma
        }
        super().__init__(
            k, 
            AutoregressiveModel, 
            component_kwargs, 
            random_seed, 
            *args, 
            **kwargs
        )

        self.p = p
        self.norm = Normal(
            loc=torch.tensor(0.0, device=self.device),
            scale=torch.tensor(1.0, device=self.device)
        )
        self.alpha = alpha
        self.beta = beta
        lam = np.asarray(lam)
        if lam.shape[0] != n_features:
            raise ValueError(f"`lam` sequence must equal `n_features` but got {len(lam)}")
        self.lam = torch.tensor(lam, dtype=torch.float32)
        self.dir_alpha = dir_alpha

    def _init_components(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor
    ) -> None:
        n_samples, time_size, n_features = X.shape
        indices = torch.randperm(n_samples, generator=self.rng)
        partitions = torch.chunk(indices, self.k)

        for i, (component, part) in enumerate(zip(self.components, partitions)):
            X_part = X[part]
            y_part = y[part]

            windows = X_part.unfold(
                dimension=1,
                size=self.p,
                step=1
            )
            rows = windows[:, :-1].flip(-1).reshape(-1, n_features * self.p)
            targets = y_part[:, self.p:].reshape(-1, 1)

            XtX = rows.T @ rows
            lam = torch.kron(self.lam, torch.eye(self.p, device=self.lam.device)).to(rows.device)
            A = XtX + lam
            b = rows.T @ targets

            coeffs = torch.linalg.solve(A, b)
            component.coeff.data.copy_(coeffs.reshape(n_features, self.p))

            residuals = targets - rows @ coeffs
            sigma = residuals.std().clamp(min=1e-3)
            component.log_sigma.data.copy_(torch.log(sigma))

            del windows, rows, XtX, lam, A, b, coeffs, sigma
        
        clear_device()
    
    def _compute_residuals(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        n_samples, time_size, _ = X.shape

        residuals = torch.zeros(
            size=(n_samples, time_size - self.p, self.k),
            device=X.device
        )
        
        for i, component in enumerate(self.components):
            preds = []
            for batch_idx in range(0, n_samples, batch_size):
                batch = X[batch_idx:batch_idx + batch_size]
                with torch.no_grad():
                    pred = component(batch)
                preds.append(pred)
            preds = torch.cat(preds, dim=0)
            residuals[:, :, i] = y[:, self.p:] - preds[:, self.p - 1:-1]
            del preds
        
        return residuals

    def _e_step(
        self, 
        residuals: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        residuals : torch.Tensor : prediction errors shape (n_samples, time_size - p + 1, k)

        Returns:
        taus : torch.Tensor : responsibilities shape (n_samples, time_size - p + 1, k)
        """
        taus = torch.zeros_like(residuals, device=residuals.device)

        for i, component in enumerate(self.components):            
            errors = residuals[:, :, i] / component.sigma
            likelihoods = self.norm.log_prob(errors) - component.log_sigma
            taus[:, :, i] = torch.log(self.weights.data[i]) + likelihoods
        
        denom = torch.logsumexp(taus, dim=2).unsqueeze(-1).expand(size=taus.shape)

        taus -= denom

        taus = torch.exp(taus)

        if torch.isnan(taus).any():
            raise RuntimeError(f"NaN found in taus\n{taus}")
        
        # Clamp to prevent small values
        taus = torch.clamp(taus, min=1e-3)
        taus = taus / taus.sum(dim=-1, keepdim=True)

        return taus

    def _m_step(
        self, 
        X: torch.Tensor,
        y: torch.Tensor, 
        taus: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Args:
        X : torch.Tensor : input features shape (n_samples, time_size, n_features)
        y : torch.Tensor; output targets shape (n_samples, time_size)
        taus : torch.Tensor : responsibilities shape (n_samples, time_size, k)
        batch_size : int : size for processing batches

        Returns:
        residuals : torch.Tensor : prediction errors shape (n_samples, time_size, k)
        """
        n_features = X.shape[2]

        windows = X.unfold(
            dimension=1,
            size=self.p,
            step=1
        )
        rows = windows[:, :-1].flip(-1).reshape(-1, n_features * self.p)
        y_flat = y[:, self.p:].reshape(-1)

        # Update components
        for i, component in enumerate(self.components):     
            taus_flat = taus[:, :, i].reshape(-1)

            sqrt_taus = torch.sqrt(taus_flat)
            X_w = rows * sqrt_taus.unsqueeze(-1)
            y_w = (y_flat * sqrt_taus).unsqueeze(-1)

            XtX = X_w.T @ X_w
            lam = torch.kron(self.lam, torch.eye(self.p, device=self.lam.device)).to(XtX.device)
            A = XtX + lam
            b = X_w.T @ y_w

            coeffs_flat = torch.linalg.solve(A, b)
            coeffs_flat = coeffs_flat.to(self.device)

            component.coeff.data.copy_(coeffs_flat.reshape(n_features, self.p))

            del XtX, X_w, y_w, lam, A, b, coeffs_flat
        
        del windows, rows

        # Compute the residuals
        residuals = self._compute_residuals(
            X,
            y,
            batch_size
        )

        # Update weights
        tau_counts = taus.sum(dim=(0,1))
        weights = (tau_counts + self.dir_alpha - 1) / (tau_counts.sum() + self.k * (self.dir_alpha - 1))
        self.weights.data.copy_(weights)

        # Update sigmas
        for i, component in enumerate(self.components):
            RSS_i = (taus[:, :, i] * residuals[:, :, i] ** 2).sum(dim=(0, 1))
            d = n_features * self.p
            lam = self.lam.clone().to(component.coeff.device)
            penalty = (lam @ (component.coeff ** 2)).sum()
            numerator = RSS_i + penalty + 2 * self.beta
            denominator = taus[:, :, i].sum(dim=(0, 1)) + d + 2 * (self.alpha + 1)
            sigma = torch.sqrt(numerator / denominator)
            component.log_sigma.data.copy_(torch.log(sigma))

        return residuals

class AutoregressiveModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        p: int, 
        rng: torch.Generator,
        init_sigma: float | None = None,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.n_features = n_features
        self.p = p
        self.coeff = nn.Parameter(
            torch.rand(size=(n_features, p), generator=rng)
        )

        if init_sigma is not None:
            log_sigma = torch.log(torch.tensor(init_sigma, dtype=torch.float32))
        else:
            log_sigma = torch.log(torch.randn(size=(1,), generator=rng))
        self.log_sigma = nn.Parameter(log_sigma)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        padded = torch.nn.functional.pad(x, (0, 0, self.p - 1, 0))
        windows = padded.unfold(dimension=1, size=self.p, step=1)

        out = torch.einsum("btfp, fp->bt", windows, self.coeff.flip(1))

        del padded, windows
        return out

    def sample(
        self,
        x: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        n_samples, window, n_features = x.shape

        context = torch.zeros(
            n_samples,
            window + horizon,
            n_features,
            device=x.device
        )
        context[:, :window, :] = x

        preds = []

        for t in range(horizon):
            mus = self(context[:, :window + t, :])
            dist = Normal(loc=mus[:, -1], scale=self.sigma.data)
            next_pred = dist.sample()
            preds.append(next_pred)
            context[:, window + t, -1] = next_pred
            del next_pred
        
        preds = torch.stack(preds, dim=1)

        del context

        return preds
    
    def log_prob(
            self,
            residuals: torch.Tensor
    ) -> torch.Tensor:
        log_prob = (
            -0.5 * torch.log(torch.tensor(2 * torch.pi)) - torch.log(self.sigma) - 
            ((residuals ** 2) / (2 * (self.sigma ** 2)))
        )
        
        if torch.isnan(log_prob).any():
            raise RuntimeError(f"nan in log_prob tensor with sigma\n{self.sigma}\n{self.log_prob}")
        
        return log_prob