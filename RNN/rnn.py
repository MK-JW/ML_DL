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

Epoch = 1
Batch_size = 50
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
y_train = data[:, -1, 0].reshape(-1, 1)
x_train = torch.tensor(x_train,dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
train_dataset = Data.TensorDataset(x_train, y_train)
# print(x_train)
# print(y_train)
# print(x_train.shape)

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True)


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

        r_out, h_n = self.rnn(x, None)
        out = self.out(r_out[:, -1, :]) # 获得最后一个时间步的输出，中间时间步输出不管
        # out = self.out(r_out)

        return out

rnn = Rnn()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=alpha)
loss_func = nn.MSELoss()


# 训练RNN网络

for epoch in range(Epoch):
    for step, (b_x, b_y) in enumerate(train_loader):
        
        # print(b_x.shape)
        # print(b_y.shape)

        output = rnn(b_x)
        # print(output.shape)
        # print(b_y)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

