__author__ = 'minjinwu'

import os 
import torch
import numpy as np
import pandas as pd
import matplotlib as plt
from MLP.linear_layer import Linear_layer
from data_process.data_normolized import Data_normalized

# 获取数据
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir,'..','world-happiness-report-2017.csv')
data = pd.read_csv(data_path)
x_train = data[['Economy..GDP.per.Capita.']].values
y_train = data[['Happiness.Score']].values



# 超参数设置
alpha = 0.001
epoch = 100
batch_size = 31

# 训练神经网络
m,n = x_train.shape
train_index = np.arange(m)
np.random.shuffle(train_index)

# 打乱数据
for start_idx in range(0, m, batch_size):
    end_idx = min(start_idx + batch_size, m)
    batch_indices = train_index[start_idx:end_idx]
    
    if len(batch_indices) < batch_size:  # 丢弃最后的小 batch
        continue

    batch_X = train_index[batch_indices]
    batch_y = y_train[batch_indices]

    print(f"Batch {start_idx // batch_size + 1}: {batch_X.shape}, {batch_y.shape}")

