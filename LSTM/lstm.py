__author__ = 'minjinwu'

# 导入所需要的模块
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

import numpy as np

# 超参数设定

Epoch = 10
Batch_size = 50
alpha = 0.01
Sample_num = 500
Time_steps = 10
Input_size= 3


# 获取所需要的数据

np.random.seed(42)
stock_price = np.cumsum(np.abs(np.random.randn(Sample_num, Time_steps)*0.1), axis=1) + 4
stock_volume = np.round(np.abs(np.random.randn(Sample_num, Time_steps)*1000) + 5000)
market_emotion = np.random.uniform(-1, 1, (Sample_num, Time_steps))
data = np.stack([stock_price, stock_volume, market_emotion], axis=-1).astype(np.float32)


# # 当数据不依赖时间顺序时

# data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
# x_train = data_train[:, :-1, :]
# y_train = data_train[:, -1, 0].reshape(-1, 1)
# # print(x_train.shape)
# x_test = data_test[:, :-1, :]
# y_test = data_test[:, -1, 0].reshape(-1, 1)
# # print(x_test.shape)
# # print(y_train)

# 当数据依赖时间顺序时
x_train = data[0:400, :-1, :]
y_train = data[0:400, -1, 0].reshape(-1, 1)
x_test = data[400:501, :-1, :]
y_test = data[400:501, -1, 0].reshape(-1, 1)
print(y_test.shape)


x_train = torch.tensor(x_train,dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
train_dataset = Data.TensorDataset(x_train, y_train)

# print(x_train.shape)
# print(y_train)
# print(x_train.shape)
print(torch.var(y_test))

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True)


# 模型的构建
class Lstm(nn.Module):

    def __init__(self):

        super(Lstm, self).__init__()

        self.lstm = nn.LSTM(

            input_size = Input_size,
            hidden_size = 64,
            num_layers = 3,
            batch_first = True,
            # dropout = 0.3,

        )

        self.out = nn.Linear(64, 1)

        pass



    def  forward(self, x):

        r_out, (h_n, c_n) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])

        return out
    
lstm = Lstm()
print(lstm)

optimizer = torch.optim.Adam(lstm.parameters(), lr=alpha)
loss_fun = nn.MSELoss()

# 训练lstm神经网络

for epoch in range(Epoch):
    
    for step, (b_x, b_y) in enumerate(train_loader):

        # print(step)
        # print(b_x.shape)
        # print(b_y.shape)
        # print(b_x)
        # print(b_y)
        # break
        output = lstm(b_x)
        # print(output.shape)  # 这里面可以看到，rnn对每一个Btach中的每一个样本的所有时间步进行循环
        # print(output)
        loss = loss_fun(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_out = lstm(x_test)
        # print(test_out.shape)
        # print('--')
        # print(y_test.shape)
        # print('--')
        # print(torch.sum((test_out - y_test)**2))
        y_mean = torch.mean(y_test)
        # print(y_mean)
        # R_2 = 1 - (torch.sum((test_out - y_test)**2) / torch.sum((y_test - y_mean)**2))  # 在数据方差很小的时候参考价值较小
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())