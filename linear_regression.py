__author__ = 'minjinwu'

import os
import pandas as pd
import numpy as np

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
y_test = data_test[['Economy..GDP.per.Capita.']].values
train_data = np.column_stack((x_train,y_train))
test_data = np.column_stack((x_test,y_test))
# print(train_data)
# print(train_data.shape)


#数据处理，数据的标准化
row, column = train_data.shape
# print(row,column)
# print(train_data[:,column-1])
# print(train_data[:,column-1].shape)
if train_data[:,column-1].std() != 0 :
    train_normalized = (train_data[:,column-1] - train_data[:,column-1].mean())/train_data[:,column-1].std()
if test_data[:,column-1].std() != 0 :
    test_normalized = (test_data[:,column-1] - test_data[:,column-1].mean())/test_data[:,column-1].std()
print(train_normalized.shape)
