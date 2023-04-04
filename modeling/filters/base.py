import numpy as np
from abc import ABC, abstractmethod

class FilterBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def filter(self, X: np.ndarray):
        pass
