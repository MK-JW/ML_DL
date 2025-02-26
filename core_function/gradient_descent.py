__author__ = 'Minjinwu'

import sys
print(sys.path)
import numpy as np
from data_process.data_normolized import Data_normolized


#执行梯度下降
class Gradient_descent: 
    def __init__(self, x_train, y_train):
        (

            x_train

        ) = Data_normolized(x_train) 


        self.x_train = x_train
        self.y_train = y_train
        

        self.sample_num, self.features_num = np.shape(self.x_train)


        def grad_descent(self, alpha, x_current):


            loss = np.array([])
            k = 0
            w_k = x_current[0:self.features_num,:]
            b_k = x_current[-1]
            y_prediction = np.dot(self.x_train, w_k) + b_k
            loss = np.append(loss, self.cost_function(y_prediction))
            dw_k = -(1/self.sample_num)*np.dot((y_prediction - y_train).T, x_train)
            db_k = -(1/self.sample_num)*np.dot((y_prediction - y_train).T, \
                            np.ones(self.sample_num, self.features_num))
            x_current = np.vstack((w_k, b_k))
            d_current = np.vstack((dw_k, db_k))

            k = 1
            x_next = x_current + alpha*d_current
            y_prediction = np.dot(self.x_train, w_k) + b_k
            loss = np.append(loss, self.cost_function(y_prediction))
            dw_k = -(1/self.sample_num)*np.dot((y_prediction - y_train).T, x_train)
            db_k = -(1/self.sample_num)*np.dot((y_prediction - y_train).T, \
                            np.ones(self.sample_num, self.features_num))
            d_current = np.vstack((dw_k, db_k))
            while np.linalg.norm(x_next - x_current)> self.tol:
                 k += 1
                 x_current = x_next
                 x_next = x_current + alpha*d_current
                 y_prediction = np.dot(self.x_train, w_k) + b_k
                 loss = np.append(loss, self.cost_function(y_prediction))
                 dw_k = -(1/self.sample_num)*np.dot((y_prediction - y_train).T, x_train)
                 db_k = -(1/self.sample_num)*np.dot((y_prediction - y_train).T, \
                            np.ones(self.sample_num, self.features_num))
                 d_current = np.vstack((dw_k, db_k))

            
            return w_k, b_k, loss, k
        

        def predict(self, alpha, x_current):

            w,b = self.gradient_descent(alpha, x_current)
            predicition = x_train*w + b
            return predicition
        

        def  cost_function(self, y_prediction):

            cost = (1/(2*self.sample_num))*np.sum((y_prediction - self.y_train)**2)

            return cost