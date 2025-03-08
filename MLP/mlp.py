__author__ = 'minjinwu'


import sys
print(sys.path)
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
epoch = 5
batch_size = 31
neurons_num = 3


# 训练神经网络

sample_num, features_num = x_train.shape
h_current = np.random.uniform(low=0, high=0.2, size = (features_num + 1, neurons_num))
o_current = np.random.uniform(low=0.05, high=0.15, size = (neurons_num + 1, 1))

for i in range(epoch):

    index = np.random.permutation(sample_num)
    x_shuffled = x_train[index]
    y_shuffled = y_train[index]

    num_batches = sample_num // batch_size

    if sample_num % batch_size != 0:  # 这里需要考虑如果有多余的数据怎么处理
        
        num_batches += 1
        giveup = True

    else: giveup = False

    for batch_index in range(num_batches):

        start = batch_index*batch_size
        end = start + batch_size
        
        if giveup == True and batch_index == num_batches - 1:
            
            continue

        x_tr = x_shuffled[start:end, :]
        y_tr = y_shuffled[start:end, :]

        prediction = Linear_layer(x_tr, y_tr).forward_propagation(h_current, o_current, neurons_num)
        h_current, o_current = Linear_layer(x_tr, y_tr).back_propagation(h_current, o_current, neurons_num, alpha)

        