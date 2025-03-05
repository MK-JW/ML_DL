__author__ = 'minjinwu'

import torch
import numpy as np
from data_process.data_normolized import Data_normalized

# 定义一个线性层，用于MLP中的线性隐藏层，主要实现以下功能：
# 1、 通过给定参数建立一个隐藏层
# 2、 可以通过输入自动得到神经元的数量


class Linear_layer:
    

    # 定义实例属性
    def __init__(self, x_train, y_train): #, neurons_num) :

        # x_train--训练集，   y_train--训练集对应的标签 

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

        # cost--损失函数，   y_predixtion--最后所得预测值

        cost = (1/(2*self.sample_num))*np.sum((y_prediction - self.y_train)**2)
         
        return cost
    

    # 获取损失函数变化
    def get_loss(self, cost):

        # loss--损失函数记录
        
        loss = np.array([])

        loss = np.append(loss, cost)

        return loss
    

    # 隐藏层输出
    def hidden_layer(self, h_current):

        # hidden_output--隐藏层对应输出，  h_current--隐藏层对应权重与偏置矩阵
        
        hidden_output = np.dot(self.x_train, h_current[0:self.features_num,:]) + h_current[-1,:]

        return hidden_output
    

    # 输出层输出
    def output_layer(self, hidden_layeroutput, o_current, neurons_num):

        # o_current--输出层对应权重与偏置向量

        prediction = np.dot(hidden_layeroutput, o_current[0:neurons_num,:]) + o_current[-1,:]

        return prediction
    

    # 输出层梯度
    def output_gradient(self, hidden_layeroutput, prediction, neurons_num):

        # d_o--输出层权重梯度，  d_ob--输出层偏置梯度

        d_o = (1/self.sample_num)*np.dot((prediction - self.y_train).T, hidden_layeroutput).T
        d_ob = (1/self.sample_num)*np.dot(((prediction - self.y_train).T, np.ones(neurons_num, 1)))
        do_current = np.vstack((d_o, d_ob))

        return do_current


    # 隐藏层梯度
    def hidden_gradient(self, o_current, prediction, neurnos_num):

        # d_h--隐藏层权重梯度，  d_hb--隐藏层偏置梯度

        d_h = (1/self.sample_num)*((prediction - self.y_train)@o_current[0:neurnos_num, :].T@self.x_train).T
        d_hb = (1/self.sample_num)*((prediction - self.y_train)@o_current[0:neurnos_num, :].T@np.ones((neurnos_num, 1))).T
        dh_current = np.vstack((d_h, d_hb))

        return dh_current
    

    # 前向传播
    def forward_propagation(self, h_current, o_current, neurons_num):

        hidden_output = self.hidden_layer(h_current)
        prediction = self.output_layer(hidden_output, o_current, neurons_num)
        cost = self.get_cost(prediction)
        loss = self.get_loss(cost)

        return prediction, loss


    # 反向传播
    def  back_propagation(self, h_current, o_current, neurons_num, alpha):

        # o_next--输出层跟新的参数，  h_next--隐藏层的下一个参数

        hidden_output = self.hidden_layer(h_current)
        prediction = self.output_layer(hidden_output, o_current, neurons_num)
        do_current = self.output_gradient(hidden_output, prediction, neurons_num)
        dh_current = self.hidden_gradient(o_current, prediction, neurons_num)
        o_next = o_current - alpha*do_current
        h_next = h_current  - alpah*dh_current

        return h_next, o_next

