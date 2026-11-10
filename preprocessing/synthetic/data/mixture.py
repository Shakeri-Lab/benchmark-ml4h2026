"""
Builds a mixture model based on ergodic components.
"""

import numpy as np
from typing import Any
from src.data.dataset import Dataset
from src.ergodic.base import BaseModel

class MixtureDataset(Dataset):
    def __init__(
            self,
            n_components: int,
            component_model: BaseModel,
            component_kwargs: dict[str, Any],
            low: float,
            high: float,
            n_features: int,
            batch_size: int,
            n_samples: int,
            time_length: int,
            random_seed: int
    ) -> None:
        super().__init__(
            batch_size,
            n_samples, 
            time_length, 
            random_seed
        )

        component_kwargs["rng"] = self.rng
        self.n_components = n_components
        self.components = self._build_components(component_model, component_kwargs)
        self.component_prior = self.rng.uniform(0, 1, size=self.n_components)
        self.component_prior /= self.component_prior.sum()

        self.low = low
        self.high = high
        self.n_features = n_features

        self.data = self._build_dataset(n_samples, time_length)

    def sample(
            self,
            features: np.ndarray
    ) -> np.ndarray:
        component = self.rng.choice(self.components, p=self.component_prior)
        batch_targets = component.forward(features)

        return batch_targets
    
    def _build_components(
            self,
            component_model: BaseModel,
            component_kwargs: dict[str, Any]
    ) -> list[BaseModel]:
        
        try:
            components = [
                component_model(**component_kwargs) for _ in range(self.n_components)
            ]
        except Exception as e:
            raise RuntimeError(f"attempted to build components but encountered Exception\n{e}")

        return components
    
    def _build_dataset(
            self,
            n_samples: int,
            time_length: int
    ) -> dict[str, np.ndarray]:
        features = self.rng.uniform(
            self.low, 
            self.high, 
            size=(
                n_samples, 
                time_length,
                self.n_features
            )
        )

        targets = []

        for idx in range(0, n_samples, self.batch_size):
            batch = features[idx:idx + self.batch_size, :, :]
            batch_targets = self.sample(batch)
            targets.append(batch_targets)
        
        targets = np.concat(targets, axis=0)

        dataset = {
            "features": features,
            "targets": targets
        }

        return dataset
