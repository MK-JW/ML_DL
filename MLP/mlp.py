__author__ = 'minjinwu'


# import sys
# print(sys.path)
import os 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLP.linear_layer import Linear_layer
from data_process.data_normolized import Data_normalized


# 获取数据
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir,'..','world-happiness-report-2017.csv')
data = pd.read_csv(data_path)
x_train = data[['Economy..GDP.per.Capita.']].values
y_train = data[['Happiness.Score']].values
print(x_train.shape, y_train.shape)


# 超参数设置
tol = 10**-6
alpha = 0.01
epoch = 5
batch_size = 31
neurons_num = 3


# 训练神经网络
loss_history = []
sample_num, features_num = x_train.shape
h_current = np.random.uniform(low=0, high=0.005, size = (features_num + 1, neurons_num))
o_current = np.random.uniform(low=0.01, high=0.02, size = (neurons_num + 1, 1))
# print(h_current)
# print(o_current)


for i in range(epoch):

    loss = np.array([])
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

        prediction, loss = Linear_layer(x_tr, y_tr).forward_propagation(h_current, o_current, neurons_num, loss)
        h_next, o_next = Linear_layer(x_tr, y_tr).back_propagation(h_current, o_current, neurons_num, alpha)

        while np.linalg.norm(h_next - h_current)>tol and np.linalg.norm(o_next - o_current)>tol:

            h_current, o_current = h_next, o_next
            prediction, loss = Linear_layer(x_tr, y_tr).forward_propagation(h_current, o_current, neurons_num, loss)
            h_next, o_next = Linear_layer(x_tr, y_tr).back_propagation(h_current, o_current, neurons_num, alpha)
        
    # loss_history = np.append(loss_history, loss)
    loss_history.append(loss)


np.array(loss_history[0]).reshape(-1,1)
# 结果的可视化
# print(x_train.shape)
# x_train = Data_normalized(x_train)
plt.figure(1)
plt.scatter(x_train, y_train, color = 'b', marker = 'o')
plt.figure(2)
plt.plot(np.arange(len(loss_history[0])), loss_history[0], color = 'b')
plt.show()
