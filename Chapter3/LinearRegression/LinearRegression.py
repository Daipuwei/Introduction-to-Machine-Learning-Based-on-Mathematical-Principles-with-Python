#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/1314:46
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : LinearRegression.py
# @Software: PyCharm

import numpy as np

"""
    这是线性回归算法Python代码，由于线性回归算法已经在第二章给出了详细介绍，
    考虑到局部加权线性回归算法主要是利用正规方程进行求解，因此线性回归算法也
    实现了正规方程求解模型参数的代码。本章的线性回归算法代码主要参照第二章的
    相关代码，但也对第二章代码进行“较小的改动”,与第二章代码有所较小的区别。
    其余BGD、SGD和MBGD三种梯度下降算法的代码实现请详见第二章的线性回归代码源文件。
"""

class LinearRegression(object):
    def __init__(self,input_data,realresult):
        """
        :param input_data: 输入数据
        :param realresult: 真实结果
        :param theta: 线性回归的参数
        """
        # 构造输入数据数组
        #input_data = np.reshape(input_data,(len(input_data)))
        self.InputData = []
        # 给每组输入数据增添常数项1
        for data in input_data:
            Data = [1.0]
            # 把input_data拓展到Data内，即把input_data的每一维数据添加到Data
            Data.extend(list(data))
            self.InputData.append(Data)
        self.InputData = np.array(self.InputData)
        #print('======')
        #print(np.shape(self.InputData))
        # 构造输入数据对应的结果
        self.Result = realresult
        #print(np.shape(self.Result))
        # 定义局部加权回归模型参数
        self.Theta = []

    def getNormalEquation(self):
        """
        这是利用正规方程计算模型参数Thetha
        """
        # 获得输入数据数组形状
        col,rol = np.shape(self.InputData.T)
        # 计算输入数据矩阵的转置
        XT = self.InputData.T
        # 计算矩阵的逆
        """
        0.001*np.eye(np.shape(self.InputData.T))是
        防止出现原始XT的行列式为0，即防止原始XT不可逆
        """
        # XTX行列式为0，即XTX不可逆是时
        if np.linalg.det(XT.dot(self.InputData)) == 0.0:
            inv = np.linalg.inv(XT.dot(self.InputData)+0.001*np.eye(col,rol))
        else:
            inv = np.linalg.inv(XT.dot(self.InputData))
        # 计算模型参数Thetha
        self.Theta = inv.dot(XT.dot(self.Result))

    def predict(self,test_data):
        """
        这是对测试数据集的线性回归预测函数
        :param test_data: 测试数据集
        """
        # 定义预测结果数组
        predict_result = []
        # 对测试数据进行遍历
        for data in test_data:
            # 预测每组data的结果
            predict_result.append(self.test(data))
        predict_result = np.array(predict_result)
        return predict_result

    def test(self,data):
        """
        这是对一组测试数据预测的函数
        :param data: 测试数据
        """
        # 对测试数据加入1维特征，以适应矩阵乘法
        tmp = [1.0]
        tmp.extend(data)
        data = np.array(tmp)
        data = data.reshape((1,len(data)))
        # 计算预测结果,计算结果形状为(1,1)，为了分析数据的方便
        # 这里只返矩阵的第一个元素
        predict_result = data.dot(self.Theta)[0][0]
        return predict_result