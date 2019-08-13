#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/1518:58
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : LinearRegression.py
# @Software: PyCharm

import numpy as np

"""
    这是线性回归算法的类，其中主要包括BGD、SGD、MBGD
    三种梯度下降算法和正则方程方法这4种实现算法。在调整
    梯度下降算法调整参数的损失函数为训练数据集的均方误差。
"""

class LinearRegression(object):
    def __init__(self,input_data,realresult,theta = None):
        """
        :param input_data: 输入数据
        :param realresult: 真实结果
        :param theta: 线性回归的参数，默认是None,即可以没有
        """
        # 获得输入数据集的形状
        row,col = np.shape(input_data)
        # 构造输入数据数组
        self.InputData = [0]*row
        # 给每组输入数据增添常数项1
        for (index,data) in enumerate(input_data):
            Data = [1.0]
            # 把input_data拓展到Data内，即把input_data的每一维数据添加到Data
            Data.extend(list(data))
            self.InputData[index] = Data
        self.InputData = np.array(self.InputData)
        # 构造输入数据对应的结果
        self.Result = realresult
        # thetha参数不为None时，利用thetha构造模型参数
        if theta is not None:
            self.Theta = theta
        else:
            # 随机生成服从标准正态分布的参数
            self.Theta = np.random.normal((col+1,1))

    def Cost(self):
        """
        这是计算损失函数的函数
        """
        # 在线性回归里的损失函数定义为真实结果与预测结果之间的均方误差
        # 首先计算输入数据的预测结果
        predict = self.InputData.dot(self.Theta).T
        # 计算真实结果与预测结果之间的均方误差
        cost = predict-self.Result.T
        cost = np.average(cost**2)
        return cost

    def BGD(self,alpha):
        """
        这是利用BGD算法进行一次迭代调整参数的函数
        :param alpha: 学习率
        """
        # 定义梯度增量数组
        gradient_increasment = []
        # 对输入的训练数据及其真实结果进行依次遍历
        for (input_data,real_result) in zip(self.InputData,self.Result):
            # 计算每组input_data的梯度增量，并放入梯度增量数组
            g = (real_result-input_data.dot(self.Theta))*input_data
            gradient_increasment.append(g)
        # 按列计算属性的平均梯度增量
        avg_g = np.average(gradient_increasment,0)
        # 改变平均梯度增量数组形状
        avg_g = avg_g.reshape((len(avg_g),1))
        # 更新参数Theta
        self.Theta = self.Theta + alpha*avg_g

    def SGD(self,alpha):
        """
        这是利用SGD算法进行一次迭代调整参数的函数
        :param alpha: 学习率
        """
        # 首先将数据集随机打乱，减小数据集顺序对参数调优的影响
        shuffle_sequence = self.Shuffle_Sequence()
        self.InputData = self.InputData[shuffle_sequence]
        self.Result = self.Result[shuffle_sequence]
        # 对训练数据集进行遍历，利用每组训练数据对参数进行调整
        for (input_data,real_result) in zip(self.InputData,self.Result):
            # 计算每组input_data的梯度增量
            g = (real_result-input_data.dot(self.Theta))*input_data
            # 调整每组input_data的梯度增量的形状
            g = g.reshape((len(g),1))
            # 更新线性回归模型参数
            self.Theta = self.Theta + alpha * g

    def Shuffle_Sequence(self):
        """
        这是在运行SGD算法或者MBGD算法之前，随机打乱后原始数据集的函数
        """
        # 首先获得训练集规模，之后按照规模生成自然数序列
        length = len(self.InputData)
        random_sequence = list(range(length))
        # 利用numpy的随机打乱函数打乱训练数据下标
        random_sequence = np.random.permutation(random_sequence)
        return random_sequence          # 返回数据集随机打乱后的数据序列

    def MBGD(self,alpha,batch_size):
        """
        这是利用MBGD算法进行一次迭代调整参数的函数
        :param alpha: 学习率
        :param batch_size: 小样本规模
        """
        # 首先将数据集随机打乱，减小数据集顺序对参数调优的影响
        shuffle_sequence = self.Shuffle_Sequence()
        self.InputData = self.InputData[shuffle_sequence]
        self.Result = self.Result[shuffle_sequence]
        # 遍历每个小批量样本
        for start in np.arange(0, len(shuffle_sequence), batch_size):
            # 判断start+batch_size是否大于数组长度，
            # 防止最后一组小样本规模可能小于batch_size的情况
            end = np.min([start + batch_size, len(shuffle_sequence)])
            # 获取训练小批量样本集及其标签
            mini_batch = shuffle_sequence[start:end]
            Mini_Train_Data = self.InputData[mini_batch]
            Mini_Train_Result = self.Result[mini_batch]
            # 定义梯度增量数组
            gradient_increasment = []
            # 对小样本训练集进行遍历，利用每个小样本的梯度增量的平均值对模型参数进行更新
            for (data,result) in zip(Mini_Train_Data,Mini_Train_Result):
                # 计算每组input_data的梯度增量，并放入梯度增量数组
                g = (result - data.dot(self.Theta)) * data
                gradient_increasment.append(g)
            # 按列计算每组小样本训练集的梯度增量的平均值，并改变其形状
            avg_g = np.average(gradient_increasment, 0)
            avg_g = avg_g.reshape((len(avg_g), 1))
            # 更新模型参数theta
            self.Theta = self.Theta + alpha * avg_g

    def getNormalEquation(self):
        """
        这是利用正规方程计算模型参数Thetha
        """
        """
          0.001*np.eye(np.shape(self.InputData.T))是
          防止出现原始XT的行列式为0，即防止原始XT不可逆
        """
        # 获得输入数据数组形状
        col,rol = np.shape(self.InputData.T)
        # 计算输入数据矩阵的转置
        XT = self.InputData.T+0.001*np.eye(col,rol)
        # 计算矩阵的逆
        inv = np.linalg.inv(XT.dot(self.InputData))
        # 计算模型参数Thetha
        self.Theta = inv.dot(XT.dot(self.Result))

    def train_BGD(self,iter,alpha):
        """
        这是利用BGD算法迭代优化的函数
        :param iter: 迭代次数
        :param alpha: 学习率
        """
        # 定义训练损失数组，记录每轮迭代的训练数据集的训练损失
        Cost = []
        # 开始进行迭代训练
        for i in range(iter):
            # 利用学习率alpha，结合BGD算法对模型进行训练
            self.BGD(alpha)
            # 记录每次迭代的训练损失
            Cost.append(self.Cost())
        Cost = np.array(Cost)
        return Cost

    def train_SGD(self,iter,alpha):
        """
        这是利用SGD算法迭代优化的函数
        :param iter: 迭代次数
        :param alpha: 学习率
        """
        # 定义训练损失数组，记录每轮迭代的训练数据集的训练损失
        Cost = []
        # 开始进行迭代训练
        for i in range(iter):
            # 利用学习率alpha，结合SGD算法对模型进行训练
            self.SGD(alpha)
            # 记录每次迭代的训练损失
            Cost.append(self.Cost())
        Cost = np.array(Cost)
        return Cost

    def train_MBGD(self,iter,batch_size,alpha):
        """
        这是利用MBGD算法迭代优化的函数
        :param iter: 迭代次数
        :param batch_size: 小样本规模
        :param alpha: 学习率
        """
        # 定义训练损失数组，记录每轮迭代的训练数据集的训练损失
        Cost = []
        # 开始进行迭代训练
        for i in range(iter):
            # 利用学习率alpha，结合MBGD算法对模型进行训练
            self.MBGD(alpha,batch_size)
            # 记录每次迭代的训练损失
            Cost.append(self.Cost())
        Cost = np.array(Cost)
        return Cost

    def test(self,test_data):
        """
        这是对测试数据集的线性回归预测函数
        :param test_data: 测试数据集
        """
        # 定义预测结果数组
        predict_result = []
        # 对测试数据进行遍历
        for data in test_data:
            # 预测每组data的结果
            predict_result.append(self.predict(data))
        predict_result = np.array(predict_result)
        return predict_result

    def predict(self,data):
        """
        这是对一组测试数据预测的函数
        :param data: 测试数据
        """
        # 对测试数据加入1维特征，以适应矩阵乘法
        tmp = [1.0]
        tmp.extend(data)
        data = np.array(tmp)
        # 计算预测结果,计算结果形状为(1,)，为了分析数据的方便
        # 这里只返矩阵的第一个元素
        predict_result = data.dot(self.Theta)[0]
        return predict_result