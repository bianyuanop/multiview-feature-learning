import numpy as np

def filter(X: np.ndarray):
    '''
    input: with shape (sample size, atlas size, ts size)
    output: filtered by bandpass filter, shape is the same with input
    '''

    return X