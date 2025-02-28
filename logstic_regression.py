__author__ = 'Minjinwu'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '.', 'bankloan.csv')
data = pd.read_csv(data_path)
data['违约'] = data['违约'].map({'是':1, '否':0})
nan_sample = data['违约'][data['违约'].isna()].index[0]
# y_train = data['违约'].iloc[0:nan_sample]
data_train = data.iloc[0:nan_sample,:]
x_train = data_train[['工龄', '地址', '收入']].values.reshape(-1,3)
y_train = data_train['违约'].values.reshape(-1,1)

