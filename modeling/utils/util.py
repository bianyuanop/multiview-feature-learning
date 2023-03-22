import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz

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

def plot_freq(ts: np.ndarray, rate: float):
    N = ts.shape[0] 
    T = 1/rate

    y = ts

    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]

    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.xlabel('Hz')

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y