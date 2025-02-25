__author__ = 'Minjinwu'

import numpy as np


def data_normolized(x_train):
    
    # 转化数据为浮点型
    x_train = np.copy(x_train).astype(float)
    
    #计算均值
    x_mean = np.mean(x_train, axis = 0)
    
    #计算方差
    x_std = np.std(x_train, axis = 0)

    #数据标准化 
    if x_train.shape[0]>1:  #这里判断行数，如果等于1均值就是他自己了

        #数据减去均值
        x_train -= x_mean

        #数据标准化
        x_std[x_std == 0] = 1
        x_train /= x_std

        #返回结果
        return x_train
