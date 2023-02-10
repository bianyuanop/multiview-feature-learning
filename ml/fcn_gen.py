import numpy as np

def gen_fcns(timeseries,  window_size=5):
    res = []
    for i in range(len(timeseries)-window_size):
        fcn = np.corrcoef(timeseries[i:i+window_size, :].T)
        res.append(fcn)
    
    return res

if __name__ == '__main__':
    arr = np.load('./output/Sub002/run1/merged.npy')
    print(arr.shape)
    fcns = gen_fcns(arr)
    print(len(fcns))
    print(fcns[0].shape)
