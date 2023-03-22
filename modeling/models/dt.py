import os
import graphviz
import pickle
import numpy as np
from .base import ModelBase
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score

class Model(ModelBase):
    def __init__(self) -> None:
        super().__init__()

        self.model = DecisionTreeClassifier()
        self.report = {}
    
    def name(self) -> str:
        return 'dt'
    
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
        
        graph_output = os.path.join(output_dir, 'tree.dot')
        tree.export_graphviz(self.model, out_file=graph_output)


    
    def load(self, target: str):
        with open(target, 'rb') as f:
            self.model = pickle.load(f)