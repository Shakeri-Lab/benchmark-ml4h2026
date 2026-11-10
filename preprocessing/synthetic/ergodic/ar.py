import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.ergodic.base import BaseModel

class AutoregressiveModel(BaseModel):
    def __init__(
            self,
            n_features: int,
            order: int,
            rng: np.random.Generator
    ) -> None:
        super().__init__()

        if order < 1:
            raise ValueError(f"`order` must be 1 or greater instead of `{order}`")
        
        self.order = order
        self.rng = rng

        loc = self.rng.uniform(-5.0, 5.0)
        scale = self.rng.uniform(1.0, 5.0)
    
        self.phi = self.rng.normal(
            loc=loc, 
            scale=scale, 
            size=(n_features, order)
        )
    
    def forward(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Args:
        x : np.ndarray : input tensor shape (batch_size, time_length, n_features)
        """
        batch, time, _ = x.shape

        out = np.zeros((batch, time))

        for t in range(min(self.order - 1, time)):
            window = x[:, t::-1, :]
            weights = self.phi[:, :t + 1]
            out[:, t] += np.einsum("blf, fl->b", window, weights)

        windows = sliding_window_view(x, window_shape=self.order, axis=1)

        out[:, self.order - 1:] += np.einsum("btfp, fp->bt", windows, self.phi[:, ::-1])

        out += self.rng.normal(size=out.shape)

        return out