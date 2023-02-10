import numpy as np

def ts_windowing(target: np.ndarray, window_size: int):
    slices_total = target.shape[0] // window_size 
    res = target.reshape((slices_total, window_size, target.shape[1]))

    return res

def get_correfs(ts: np.ndarray, altas_len: int):
    res = np.zeros((ts.shape[0], altas_len, altas_len))
    for i in range(ts.shape[0]):
        sample = ts[i, :, :]
        res[i, :, :] = np.corrcoef(sample.T)
    
    return res
