import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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


df_labels = pd.read_csv('./data/labels.csv')
cate_enc = LabelEncoder()
cate_enc.fit(df_labels.iloc[:, 1].to_numpy())
print(df_labels.iloc[:, 1].to_numpy())

base_dir = './output'
subjects = os.listdir(base_dir)
runs = [f'run{i}' for i in range(1, 5)]
WINDOW_SIZE = 19
ATLAS_COUNT = 164

output_dir = 'fcns'

sample_ts = np.load(os.path.join(base_dir, subjects[0], runs[0], 'merged.npy'))
slices = ts_windowing(sample_ts, WINDOW_SIZE)
fcns = get_correfs(slices, ATLAS_COUNT)

Xs = np.zeros((len(subjects)*len(runs)*slices.shape[0], ATLAS_COUNT, ATLAS_COUNT))
ys = np.zeros((len(subjects)*len(runs)*slices.shape[0]))

subjects.sort()

for i in range(len(subjects)):
    for j in range(len(runs)):
        sub = subjects[i]
        run = runs[j]

        ts = np.load(os.path.join(base_dir, sub, run, 'merged.npy'))
        slices = ts_windowing(ts, WINDOW_SIZE)
        fcns = get_correfs(slices, ATLAS_COUNT)


        X = fcns
        y = cate_enc.transform(df_labels.loc[:, run].to_numpy()) 

        # print(X.shape)

        start = i*4*slices.shape[0] + j*slices.shape[0]
        end = i*4*slices.shape[0] + (j+1)*slices.shape[0]

        print(f'processing {sub}-{run} stored at {start}:{end}')
        

        Xs[start:end, :, :] = X
        ys[start:end] = y

        # Xs.append(Xs)
        # ys.append(ys)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'Xs.npy'), 'wb') as f:
    np.save(f, Xs)
        

with open(os.path.join(output_dir, 'ys.npy'), 'wb') as f:
    np.save(f, ys)