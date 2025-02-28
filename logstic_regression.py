__author__ = 'Minjinwu'

import os
import torch
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
x_train = data_train[['工龄', '地址', '收入', '负债率', '信用卡负债', '其他负债']].values.reshape(-1,6)
y_train = data_train['违约'].values.reshape(-1,1)


alpha = 0.001
tol = 10**-6
loss = np.array([])
m,n = x_train.shape
x_current = np.zeros((n+1,1)).reshape(-1,1)
z = np.dot(x_train, x_current[0:n,:]) + x_current[-1,:]
y_pr = torch.sigmoid(torch.tensor(z))
y_pr = np.array(y_pr)
# print(y_pr)

cost = -(1/m)*(np.dot(y_train.T, np.log(y_pr)) + np.dot((1 - y_train).T, np.log(1 - y_pr)))
dw = -(1/m)*np.dot((y_pr - y_train).T, x_train).T
db = -(1/m)*np.dot((y_pr - y_train).T, np.ones((m,1))).T
d_current = np.vstack((dw, db))
# print(d_current)

loss = np.append(loss, cost)
x_next = x_current + alpha*d_current
while np.linalg.norm(x_next - x_current)> tol:

    x_current = x_next
    z = np.dot(x_train, x_current[0:n,:]) + x_current[-1,:]
    y_pr = torch.sigmoid(torch.tensor(z))
    y_pr = np.array(y_pr)
    cost = -(1/m)*(np.dot(y_train.T, np.log(y_pr)) + np.dot((1 - y_train).T, np.log(1 - y_pr)))
    dw = -(1/m)*np.dot((y_pr - y_train).T, x_train).T
    db = -(1/m)*np.dot((y_pr - y_train).T, np.ones((m, 1))).T
    d_current = np.vstack((dw, db))

    loss = np.append(loss, cost)
    x_next = x_current + alpha*d_current

y_pr = y_pr> 0.5
y_pr = y_pr.astype(int)
print(np.hstack((y_train, y_pr)))