import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        pass