#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/1314:40
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : LocalWeightedLinearRegression.py
# @Software: PyCharm

"""
    这是利用正规方程求解局部加权线性回归模型参数的Python代码
"""

import numpy as np

class LocalWeightedLinearRegression(object):
    def __init__(self,train_data,train_result):
        """
        :param train_data: 输入训练数据
        :param train_result: 训练数据真实结果
        """
        # 获得输入数据集的形状
        row, col = np.shape(train_data)
        # 构造输入数据数组
        self.Train_Data = [0] * row
        # 给每组输入数据增添常数项1
        for (index, data) in enumerate(train_data):
            Data = [1.0]
            # 把每组data拓展到Data内，即把每组data的每一维数据依次添加到Data
            Data.extend(list(data))
            self.Train_Data[index] = Data
        self.Train_Data = np.array(self.Train_Data)
        # 构造输入数据对应的结果
        self.Train_Result = train_result
        # 定义数据权重
        self.weight = np.zeros((row,row))
        # 定义局部加权回归模型参数
        self.Theta = []

    def Gaussian_Weight(self,data,k):
        """
        这是计算测试权重的函数
        :param data: 输入数据
        :param k: 带宽系数
        """
        # data的数据类型是np.array，那么利用dot方法
        # 进行矩阵运算的结果是矩阵，哪怕只有一个元素
        sum = np.sum(data*data)
        return np.exp(sum/(-2.0*k**2))

    def predict_NormalEquation(self,test_data,k):
        """
        这是利用正规方程对测试数据集的局部加权线性回归预测函数
        :param test_data: 测试数据集
        :param k: 带宽系数
        """
        # 对测试数据集全加入一维1，以适应矩阵乘法
        data = []
        for test in test_data:
            # 对测试数据加入1维特征，以适应矩阵乘法
            tmp = [1.0]
            tmp.extend(test)
            data.append(tmp)
        test_data = np.array(data)
        # 计算test_data与训练数据集之间的权重矩阵
        for (index, train_data) in enumerate(self.Train_Data):
            diff = test_data-self.Train_Data[index]
            self.weight[index, index] = self.Gaussian_Weight(diff, k)
        # 计算XTWX
        XTWX = self.Train_Data.T.dot(self.weight).dot(self.Train_Data)
        """
          0.001*np.eye(np.shape(self.Train_Data.T))是
          防止出现原始XT的行列式为0，即防止原始XT不可逆
       """
        # 获得输入数据数组形状
        row, col = np.shape(self.Train_Data)
        # 若XTWX的行列式为0，即XTWX不可逆，对XTWX进行数学处理
        if np.linalg.det(XTWX) == 0.0:
            XTWX = XTWX + 0.001 * np.eye(col, col)
        # 计算矩阵的逆
        inv = np.linalg.inv(XTWX)
        # 计算模型参数Thetha
        XTWY = self.Train_Data.T.dot(self.weight).\
            dot(np.reshape(self.Train_Result,(len(self.Train_Result),1)))
        self.Theta = inv.dot(XTWY)
        # 对测试数据test_data进行预测
        predict_result = test_data.dot(self.Theta).T[0]
        return predict_result

