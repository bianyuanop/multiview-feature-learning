import numpy  as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class LassoSelector:
    def __init__(self, atlas_cares: np.ndarray) -> None:
        self.atlas_cares = atlas_cares
    
    def transform(self, X: np.ndarray):
        # in case there is no feature is selected
        if self.atlas_cares.shape[0] != 0:
            return X[:, self.atlas_cares, :] 

        return X

def process(X: np.ndarray, y: np.ndarray):
    '''
    input: time series in form (sample id, atlas id, time)
    output: a selector that transforms X into selected features
    '''
    atlas_len = X.shape[1]
    coefs = np.zeros((X.shape[0], atlas_len, atlas_len))

    for i in range(X.shape[0]):
        coefs[i, :, :] = np.corrcoef(X[i, :, :])
    
    X_lasso = coefs.reshape(coefs.shape[0], -1)
    y_lasso = y

    
    pipeline = Pipeline([
        ('model', Lasso())
    ])

    search = GridSearchCV(pipeline,
                        {'model__alpha':np.arange(0.1,10,0.1)},
                        cv = 5, scoring="neg_mean_squared_error",verbose=3
                        )

    search.fit(X_lasso, y_lasso)

    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)

    significant_conns = np.where(importance > 0)[0]

    atlas_cares = significant_conns // atlas_len

    selector = LassoSelector(atlas_cares)

    return selector
