__author__ = 'Minjinwu'

# import sys
# print(sys.path)
import numpy as np
from data_process.data_normolized import Data_normalized


#执行梯度下降
class Gradient_descent: 

    def __init__(self, x_train, y_train):

        (

            x_train

        ) = Data_normalized(x_train) 


        self.x_train = x_train
        self.y_train = y_train
        
        self.sample_num, self.features_num = self.x_train.shape


    def get_cost(self, y_prediciton):

        cost = (1/(2*self.sample_num))*np.sum(((y_prediciton - self.y_train)**2))

        return cost
    

    def get_prediction(self, w, b):

        y_prediction = np.dot(self.x_train, w) + b

        return y_prediction
    

    def get_gradient(self, y_prediction):
        
        dw = (1/self.sample_num)*np.dot((y_prediction - self.y_train).T, self.x_train).T
        db = (1/self.sample_num)*np.dot((y_prediction - self.y_train).T, np.ones((self.sample_num, 1)))
        d_current = np.vstack((dw, db))

        return d_current
    

    def get_loss(self, cost):

        loss = np.array([])
        
        loss = np.append(loss, cost)

        return loss


    def upgrade_parameters(self, alpha, x_current, d_current):

        x_next = x_current + alpha*d_current

        return x_next
    




        

