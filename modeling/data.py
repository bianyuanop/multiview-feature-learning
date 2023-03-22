import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

df_labels = pd.read_csv('./data/labels.csv')
df_labels

one_hot_enc = OneHotEncoder(handle_unknown='ignore')
one_hot_enc.fit_transform(df_labels.iloc[:, 1].to_numpy().reshape(-1, 1))

cate_enc = LabelEncoder()
cate_enc.fit_transform(df_labels.iloc[:, 1].to_numpy())


base_dir = './output/'
subjects = os.listdir(base_dir)
runs = [f'run{i}' for i in range(1, 5)]
WINDOW_SIZE = 19

print('='*10 + 'PREPARING DATA' + '='*10)
print('subjects: ', subjects)
print('runs: ', runs)

from utils.util import ts_windowing

Xs = []
ys = []
for sub in subjects:
    for run in runs:
        ts = np.load(f'./output/{sub}/{run}/merged.npy')
        ts_windowed = ts_windowing(ts, WINDOW_SIZE)

        Xs.append(ts_windowed)
        ys.append(cate_enc.transform(df_labels.loc[:, run].to_numpy()))
    
    
sample_shape = Xs[0].shape

Xs = np.array(Xs).reshape(len(subjects)*len(runs)*sample_shape[0], sample_shape[1], sample_shape[2]).transpose(0, 2, 1)
ys = np.array(ys).reshape(len(subjects)*len(runs)*sample_shape[0])

X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.1, stratify=ys)
print('='*10 + 'DATA PREPARED' + '='*10)