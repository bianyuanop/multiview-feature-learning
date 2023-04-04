import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

def filter(X: np.ndarray, lowcut=0.015, highcut=0.1, interval=2, order=6):
    '''
    input: with shape (sample size, atlas size, ts size)
    output: filtered by bandpass filter, shape is the same with input
    '''
    fs = 1/interval

    def butter_bandpass(lowcut, highcut, fs, order=order):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=order):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # for order in [3, 6, 9]:
    #     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #     w, h = freqz(b, a, fs=fs, worN=2000)
    #     plt.plot(w, abs(h), label="order = %d" % order)
    
    filtered = np.zeros_like(X)

    for i in range(filtered.shape[0]):
        sample = X[i, :, :]
        filtered[i, :, :] = butter_bandpass_filter(sample, lowcut, highcut, fs)

    return filtered