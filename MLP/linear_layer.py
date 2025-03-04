__author__ = 'minjinwu'

import torch
import numpy as np
import pandas as pd
from data_process.data_normolized import Data_normalized

# 定义一个线性层，用于MLP中的线性隐藏层，主要实现以下功能：
# 1、 通过给定参数建立一个隐藏层
# 2、 可以通过输入自动得到神经元的数量

class Linear_layer:
    
    # 定义实例属性
    def __init__(self, x_train, y_train): #, neurons_num) :
        (
            x_train

        ) = Data_normalized(x_train)

        self.x_train = x_train
        self.y_train = y_train

        self.sample_num, self.features_num = x_train.shape
        # self.num = neurons_num

        pass

    # 损失函数    
    def get_cost(self, y_prediction):

        cost = (1/(2*self.sample_num))*np.sum((y_prediction - self.y_train)**2)
         
        return cost
    
    # 获取损失函数变化
    def get_loss(self, cost):
        
        loss = np.array([])

        loss = np.append(loss, cost)

        return loss
    
    # 隐藏层输出
    def hidden_layer(self, h_current):
        
        hidden_output = np.dot(self.x_train, h_current[0:self.features_num,:]) + h_current[-1,:]

        return hidden_output
    
    # 输出层输出
    def output_layer(self, hidden_layeroutput, o_current, neurons_num):

        prediction = np.dot(hidden_layeroutput, o_current[0:neurons_num,:]) + o_current[-1,:]

        return prediction
    
    # 输出层梯度
    def output_gradient(self, hidden_layeroutput, prediction, neurons_num):

        d_o = (1/self.sample_num)*np.dot((prediction - self.y_train).T, hidden_layeroutput).T
        d_ob = (1/self.sample_num)*np.dot(((prediction - self.y_train).T, np.ones(neurons_num, 1)))
        do_current = np.vstack((d_o, d_ob))

        return do_current

    # 隐藏层梯度
    def hidden_gradient(self, o_current, prediction, neurnos_num):

        d_h = (1/self.sample_num)*((prediction - self.y_train)@o_current[0:neurnos_num, :]@self.x_train)
        d_hb = 

        return d_w