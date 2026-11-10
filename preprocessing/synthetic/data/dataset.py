"""
Base class for building a synthetic dataset.
"""

import numpy as np
from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(
            self,
            batch_size: int,
            n_samples: int,
            time_length: int,
            random_seed: int
    ) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.time_length = time_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def sample(
            self,
            features: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _build_dataset(
            self,
            n_samples: int,
            time_length: int
    ) -> dict[str, np.ndarray]:
        pass

    def len(
            self
    ) -> int:
        return self.n_samples