from abc import ABC, abstractmethod
import numpy as np

class DataBase(ABC):
    def __init__(self) -> None:
        '''
        you should initalize the class here for preparing X and y
        '''
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    @abstractmethod
    def y(self) -> np.ndarray:
        pass
