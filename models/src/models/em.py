import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from src.utils.device import set_device, clear_device

class EMModel(ABC, nn.Module):
    def __init__(
        self, 
        k: int,
        component: nn.Module,
        component_kwargs: dict[str, Any],
        random_seed: int,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.device = set_device()
        self.rng = torch.Generator().manual_seed(random_seed)
        component_kwargs["rng"] = self.rng
        self.k = k
        self.components = nn.ModuleList(
            [
                component(**component_kwargs) for _ in range(k)
            ]
        )

        self.weights = nn.Parameter(self._init_prior())

        self.data_stats = None

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        batch_size, time_size, _ = x.shape
    
        preds = []
        
        for component in self.components:
            preds.append(component(x))
        
        preds = torch.stack(preds, dim=-1)

        #weights = torch.expand_copy(self.weights, size=(batch_size, time_size, self.k))

        out = preds * self.weights

        del preds

        return out
    
    def predict(
        self,
        x: torch.Tensor,
        horizon: int = 1,
    ) -> torch.Tensor:
        """
        A prediction function for taking the weighted average across mixture components.

        Horizon controls whether we do autoregressive rollout or single (teacher-forced) prediction.

        Output preds tensor assumes all of input x tensor dim 1 as context for horizon forecast when 
        horizon is > 1.

        If doing multivariate to univariate forecasting, we expect the last input feature to be the 
        target.

        Args:
        x : torch.Tensor : input tensor with shape (n_samples, window, n_features)
        horizon : int : the horizon forecast, default 1 does teacher-forcing next item prediction

        Returns:
        preds : torch.Tensor : output tensor with shape (n_samples, window - 1 + horizon)
        """
        if self.data_stats is None:
            raise RuntimeError("data_stats not initialized; run train first")

        with torch.no_grad():
            x = (x - self.data_stats["x_mean"]) / self.data_stats["x_std"]

            if horizon > 1:
                n_samples, window, n_features = x.shape
                preds = []

                context = torch.zeros(
                    n_samples, 
                    window + horizon, 
                    n_features, 
                    device=x.device
                )
                context[:, :window, :] = x

                for t in range(horizon):
                    pred = self(context[:, :window + t, :]).sum(dim=-1)
                    next_pred = pred[:, -1].clone()
                    preds.append(next_pred)
                    context[:, t + window, -1] = next_pred
                    del pred
                
                preds = torch.stack(preds, dim=1)
                del context
            else:
                preds = self(x).sum(dim=-1)

            preds = preds * self.data_stats["y_std"] + self.data_stats["y_mean"]

        return preds
    
    def predict_proba(
        self,
        x: torch.Tensor,
        horizon: int,
        n_samples: int = 128
    ) -> dict[str, torch.Tensor]:
        """
        Args:
        x : torch.Tensor : input tensor with shape (n_samples, window, n_features)
        horizon : int : the horizon forecast, default 1 does teacher-forcing next item prediction
        """
        if self.data_stats is None:
            raise RuntimeError("data_stats not initialized; run train first")

        with torch.no_grad():
            x = (x - self.data_stats["x_mean"]) / self.data_stats["x_std"]
            try:
                samples = []
                for _ in range(n_samples):
                    preds = self._sample(x, horizon=horizon)
                    preds = preds * self.data_stats["y_std"] + self.data_stats["y_mean"]
                    samples.append(preds)
                samples = torch.stack(samples, dim=-1)
            except Exception as e:
                raise RuntimeError(f"attempted to sample from distribution but got error\n {e}")
        
        mean = samples.mean(dim=-1)
        std = samples.std(dim=-1)

        return {
            "mean": mean,
            "std": std,
            "lower": mean - 2 * std,
            "upper": mean + 2 * std
        }

    
    def log_likelihood(
        self,
        residuals: torch.Tensor
    ) -> torch.Tensor:
        log_probs = []

        for i, component in enumerate(self.components):
            try:
                log_probs.append(component.log_prob(residuals[:, :, i]))
            except Exception as e:
                raise RuntimeError(f"Attempted to compute log probability on component {i} with error:\n{e}")
        
        log_probs = torch.stack(log_probs, dim=-1)

        weighted_log_probs = torch.log(self.weights.data) + log_probs

        log_mix = torch.logsumexp(weighted_log_probs, dim=-1)

        return log_mix.sum()
    
    def train(
        self,
        dataset: dict[str, torch.Tensor],
        batch_size: int = 128,
        tol: float = 1e-5,
        max_iter: int = 100
    ) -> None:
        X = dataset["features"].to(self.device)
        y = dataset["targets"].to(self.device)
        self.to(self.device)

        # Standardize to handle
        self.data_stats = {
            "x_mean": X.mean(dim=(0, 1), keepdim=True),
            "x_std": X.std(dim=(0, 1), keepdim=True),
            "y_mean": y.mean(),
            "y_std": y.std()
        }
        X = (X - self.data_stats["x_mean"]) / self.data_stats["x_std"]
        y = (y - self.data_stats["y_mean"]) / self.data_stats["y_std"]

        prev_likelihood = None

        # Initialize components from data
        self._init_components(X, y)
        print("initialized components")

        # Precompute residuals for first e step
        residuals = self._compute_residuals(
            X,
            y,
            batch_size
        )

        with torch.no_grad():
            for step in range(max_iter):
                tau = self._e_step(residuals)
                residuals = self._m_step(
                    X,
                    y,
                    tau,
                    batch_size=batch_size
                )

                n_obs = residuals.shape[0] * residuals.shape[1]
                current_likelihood = self.log_likelihood(residuals).item() / n_obs

                print(f"epoch {step} negative log likelihood {-current_likelihood:.4f}")

                if prev_likelihood is not None:
                    rel_change = (
                        np.abs(prev_likelihood - current_likelihood) / 
                        np.abs(prev_likelihood)
                    )

                    if rel_change < tol:
                        print(f"relative change below tolerance {tol}")
                        break

                prev_likelihood = current_likelihood

                clear_device()

    def _init_prior(
        self
    ) -> torch.Tensor:
        weights = torch.rand(self.k, generator=self.rng)
        weights /= weights.sum()

        return weights
    
    def _sample(
        self,
        x: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        idx = torch.multinomial(self.weights.data, 1)
        sample = self.components[idx].sample(x, horizon)
        return sample

    @abstractmethod
    def _init_components(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        pass
    
    @abstractmethod
    def _compute_residuals(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def _e_step(
        self,
        residuals: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def _m_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        taus: torch.Tensor,
        batch_size: int
    ) -> None:
        pass