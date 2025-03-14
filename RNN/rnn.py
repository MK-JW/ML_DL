__author__ = 'minjinwu'


# 导入所需要的库

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

import numpy as np 


# 设定超参数

epoch = 1
batch_size = 50
alpha = 0.001
Sample_num = 500
Time_steps = 10
Input_size= 3


# 生成所需要的数据

np.random.seed(42)
stock_price = np.cumsum(np.random.randn(Sample_num, Time_steps)*0.1, axis=1) + 4
stock_volume = np.round(np.abs(np.random.randn(Sample_num, Time_steps)*1000) + 5000)
market_emotion = np.random.uniform(-1, 1, (Sample_num, Time_steps))
data = np.stack([stock_price, stock_volume, market_emotion], axis=-1).astype(np.float32)
x_train = data[:, :-1, :]
y_train = data[:, -1, 0].reshape(-1,1)
print(x_train.shape)


# 搭建RNN模型

class Rnn(nn.Module):

    def  __init__(self):
        super(Rnn, self).__init__()
        
        self.rnn = nn.RNN(
            input_size = Input_size,
            hidden_size = 64,
            num_layers = 3,
            batch_first = True,
        )

        self.out = nn.Linear(64, 1)

        pass

    def  forward(self, x):

        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])

        pass

rnn = Rnn()
print(rnn)

