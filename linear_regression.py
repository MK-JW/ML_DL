__author__ = 'minjinwu'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__) 
data_path = os.path.join(current_dir,'.','world-happiness-report-2017.csv')
# print(data_path)
data = pd.read_csv(data_path)
data_train = data.sample(frac=0.7)
data_test = data.drop(data_train.index)
# print(data_train.index)
# print(data_train)
x_train = data_train[['Economy..GDP.per.Capita.']].values
y_train = data_train[['Happiness.Score']].values
x_test = data_test[['Economy..GDP.per.Capita.']].values
y_test = data_test[['Happiness.Score']].values
train_data = np.column_stack((x_train,y_train))
test_data = np.column_stack((x_test,y_test))
# print(test_data)
# print(train_data.shape)


#数据处理，数据的标准化
row, column = train_data.shape
# print(row,column)
# print(train_data[:,1:column])
# print(train_data[:,1:column].shape)
if train_data[:,1:column].std() != 0 :
    train_normalized = (train_data[:,1:column] - train_data[:,1:column].mean(axis=0))/train_data[:,1:column].std(axis=0)
    train_data[:,1:column] = train_normalized
if test_data[:,1:column].std() != 0 :
    test_normalized = (test_data[:,1:column] - test_data[:,1:column].mean(axis=0))/test_data[:,1:column].std(axis=0)
    test_data[:,1:column] = test_normalized
# print(test_data)
# print(train_data.shape)


# 执行梯度下降
k = 0  #代表迭代数
m, n = x_train.shape
w_k = 0
b_k = 0
alpha = 0.5
tol = 10**-6
loss = np.array([])
y_prediction = w_k*x_train + b_k
cost = (1/(2*m))*np.sum((y_prediction - y_train)**2)
dw_k = -(1/m)*np.dot((y_prediction - y_train).T, x_train)
db_k = -(1/m)*np.dot((y_prediction - y_train).T, np.ones((m,n)))
x_current = np.array(([w_k],[b_k]))  #.reshape(2,1)
d_current = np.row_stack((dw_k, db_k))
loss = np.append(loss,cost)
# print(x_current)
# print(x_current.shape)
# print(d_current)
# print(d_current.shape)
k = 1
x_next = x_current + alpha*d_current
w_k = x_next[0,0]
b_k  = x_next[1,0]
y_prediction = w_k*x_train + b_k
cost = (1/(2*m))*np.sum((y_prediction - y_train)**2)
dw_k = -(1/m)*np.dot((y_prediction - y_train).T, x_train)
db_k = -(1/m)*np.dot((y_prediction - y_train).T, np.ones((m,n)))
d_current = np.row_stack((dw_k, db_k))
loss = np.append(loss,cost)
while np.linalg.norm(x_next - x_current)> tol :
    x_current = x_next
    x_next = x_current + alpha*d_current
    w_k = x_next[0,0]
    b_k  = x_next[1,0]
    y_prediction = w_k*x_train + b_k
    cost = (1/(2*m))*np.sum((y_prediction - y_train)**2)
    dw_k = -(1/m)*np.dot((y_prediction - y_train).T, x_train)
    db_k = -(1/m)*np.dot((y_prediction - y_train).T, np.ones((m,n)))
    d_current = np.row_stack((dw_k, db_k))
    loss = np.append(loss,cost)
print(loss.shape)
loss = loss.reshape(-1,1)
print(loss.shape)
print(type(loss))
q,p = loss.shape
print(w_k,b_k)


print(x_train.shape, y_train.shape)
# 执行绘图
if column == 2:
    plt.figure(1)
    plt.scatter(x_train, y_train, color = 'r', marker = 'o')
    plt.xlabel('x_train')
    plt.ylabel('y_train')
    plt.plot(np.arange(x_train.min(), x_train.max(), 0.1), 
             w_k*np.arange(x_train.min(), x_train.max(), 0.1) + b_k, color = 'k')
    plt.figure(2)
    plt.scatter(x_test, y_test, color = 'b', marker = 'o')
    plt.xlabel('x_test')
    plt.ylabel('y_test')
    plt.plot(np.arange(x_test.min(), x_test.max(), 0.1),
              w_k*np.arange(x_test.min(), x_test.max(), 0.1) + b_k, color = 'k')
    plt.figure(3)
    plt.plot(np.arange(q).reshape(q,1), loss, color = 'b')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()