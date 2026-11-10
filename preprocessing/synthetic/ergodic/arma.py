import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.ergodic.base import BaseModel

class ARMA(BaseModel):
    def __init__(
            self,
            n_features: int,
            time_length: int,
            p: int,
            q: int,
            rng: np.random.Generator
    ) -> None:
        super().__init__()

        self.p = p
        self.q = q
        self.time_length = time_length
        self.rng = rng

        if p < 1 or q < 1:
            raise ValueError(f"`p` and `q` must be 1 or greater instead of {p} and {q}")
        
        loc = self.rng.uniform(-5.0, 5.0)
        scale = self.rng.uniform(1.0, 5.0)

        self.phi = self.rng.normal(
            loc=loc,
            scale=scale,
            size=(n_features, p)
        )

        loc = self.rng.uniform(-5.0, 5.0)
        scale = self.rng.uniform(1.0, 5.0)

        self.theta = self.rng.normal(
            loc=loc,
            scale=scale,
            size=(q,)
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

        for t in range(min(self.p - 1, time)):
            window = x[:, t::-1, :]
            weights = self.phi[:, :t + 1]
            out[:, t] += np.einsum("blf, fl->b", window, weights)

        windows = sliding_window_view(x, window_shape=self.p, axis=1)

        out[:, self.p - 1:] += np.einsum("btfp, fp->bt", windows, self.phi[:, ::-1])

        errors = self.rng.normal(size=(x.shape[0], time + self.q))

        out += errors[:, self.q:]

        ma_out = np.zeros((batch, time))

        for t in range(min(self.q - 1, time)):
            window = errors[:, t::-1]
            weights = self.theta[:t + 1]
            ma_out[:, t] += np.einsum("bl, l->b", window, weights)

        ma_windows = sliding_window_view(errors, window_shape=self.q, axis=1)

        ma_out[:, self.q - 1:] += np.einsum(
            "btq, q->bt", 
            ma_windows[:, :time - self.q + 1, :], 
            self.theta[::-1]
        )

        out += ma_out

        return out