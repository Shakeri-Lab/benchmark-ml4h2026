import numpy as np
from src.ergodic.base import BaseModel

class OrnsteinUhlenbeck(BaseModel):
    def __init__(
            self,
            n_features: int,
            rng: np.random.Generator
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.rng = rng

        self.beta = self.rng.uniform(
            -5.0, 
            5.0, 
            size=(n_features,)
        )

        self.theta = self.rng.uniform(0.01, 2.0)

        self.sigma = self.rng.uniform(0.01, 2.0)
    
    def forward(
            self,
            x: np.ndarray
    ) -> None:
        """
        Args:
        x : np.ndarray : input tensor shape (batch_size, time_length, n_features)
        """

        batch_size, time_length, n_features = x.shape

        out = np.zeros((batch_size, time_length))

        for t in range(1, time_length):
            target = x[:, t, :] @ self.beta
            dW = self.rng.normal(size=(batch_size,))
            out[:, t] = (
                out[:, t-1]
                + self.theta * (target - out[:, t-1])
                + self.sigma * dW
            )
        
        return out