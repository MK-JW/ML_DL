__author__  = 'Minjinwu'


# import sys
# print(sys.path)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.data_normolized import Data_normolized
from core_function.gradient_descent import Gradient_descent


current_dir = os.path.dirname(__file__) 
data_path = os.path.join(current_dir,'.','world-happiness-report-2017.csv')
data = pd.read_csv(data_path)
x_train = data[['Economy..GDP.per.Capita.']].values
y_train = data[['Happiness.Score']].values

k = 0
alpha = 0.1
tol = 1*10**-6
m,n = x_train.shape
x_train = Data_normolized(x_train)
x_current = np.zeros((n+1, 1))
y_prediction = Gradient_descent(x_train, y_train).get_prediction(x_current[0:n,:], x_current[-1, :])
cost = Gradient_descent(x_train, y_train).get_cost(y_prediction)
loss = Gradient_descent(x_train, y_train).get_loss(cost)
d_current = -Gradient_descent(x_train, y_train).get_gradient(y_prediction)
x_next = Gradient_descent(x_train, y_train).upgrade_parameters(alpha, x_current, d_current)

while np.linalg.norm(x_next - x_current)> tol:

    k += 1
    x_current = x_next
    y_prediction = Gradient_descent(x_train, y_train).get_prediction(x_current[0:n,:], x_current[-1, :])
    cost = Gradient_descent(x_train, y_train).get_cost(y_prediction)
    loss = Gradient_descent(x_train, y_train).get_loss(cost)
    d_current = -Gradient_descent(x_train, y_train).get_gradient(y_prediction)
    x_next = Gradient_descent(x_train, y_train).upgrade_parameters(alpha, x_current, d_current)

y_p = Gradient_descent(x_train, y_train).get_prediction(x_next[0:n,:], x_next[-1,:])
print(y_p.shape)


plt.figure(1)
plt.scatter(x_train, y_train, color = 'r', marker = 'o')
plt.plot(np.arange(x_train.min(), x_train.max(), 0.1).reshape(-1,1), np.arange(x_train.min(), x_train.max(), 0.1).reshape(-1,1)*x_next[0:n,:] + x_next[-1,:], color = 'k')
plt.show()
