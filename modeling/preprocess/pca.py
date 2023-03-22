import numpy as np
from sklearn.decomposition import PCA

class PCASelector:
    def __init__(self, pca: PCA) -> None:
        self.pca = pca
    
    def transform(self, X: np.ndarray):
        X_pca = X.reshape(X.shape[0], -1)
        return self.pca.transform(X_pca)

def process(X: np.ndarray, n_components=500, *args):
    pca = PCA(n_components=n_components) 

    X_pca = X.reshape(X.shape[0], -1)
    pca.fit(X_pca)

    selector = PCASelector(pca)

    return selector