import os
import pickle
import numpy as np
from sklearn.svm import SVC
from .base import ModelBase
from sklearn.metrics import classification_report, accuracy_score


class Model(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC()
        self.report = {}
    
    def name(self) -> str:
        return 'svm'

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_and_report(self, X: np.ndarray, y: np.ndarray):
        ypred = self.predict(X)

        self.report['accuracy'] = str(accuracy_score(y, ypred))
        self.report['classification report'] = classification_report(y, ypred)
    
    def save(self, output_dir: str):
        with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
            for key, report in self.report.items():
                f.write(key + '\n')
                f.write(report + '\n')
            
        with open(os.path.join(output_dir, 'model.bin'), 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, target: str):
        with open(target, 'rb') as f:
            self.model = pickle.load(f)

def get_model(X: np.ndarray, y: np.ndarray):
    model = SVC()
    print(X.shape, y.shape)
    model.fit(X, y)

    return model
    