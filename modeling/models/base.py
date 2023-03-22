from abc import ABC, abstractmethod
import numpy as np

class ModelBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_and_report(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def load(self, target: str):
        pass

    @abstractmethod
    def save(self, output_dir: str):
        '''
        This method should save model and report 
        '''
        pass
    