#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/3 17:30
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : SoftmaxRegression.py
# @Software: PyCharm

'''
    这是Softmax回归类代码，实现了BGD、SGD和MBGD梯度下降算法
    的参数优化算法
'''

import numpy as np

class SoftmaxRegression(object):
    def __init__(self,Train_Data,Train_Label,theta = None):
        """
        这是Softmax回归算法的初始化函数
        :param Train_Data: 训练数据集,类型为numpy.ndarray
        :param Train_Label: 训练数据集标签
        :param theta: 训练数据集标签,类型为numpy.ndarray，默认为None，即可以没有
        """
        self.Train_Data = []    # 定义训练数据集
        for train_data in Train_Data:
            data = [1.0]
            # 把data拓展到Data内，即把data的每一维数据添加到data
            data.extend(list(train_data))
            self.Train_Data.append(data)
        self.Train_Data = np.array(self.Train_Data)
        # 定义训练标签集
        # 若标签数组是2维数组，即标签是one-hot编码
        if len(np.shape(Train_Label)) == 2:
            self.Train_Label = Train_Label
        else:
            # 若标签不是one-hot编码，那么必须进行转换
            self.Train_Label = self.Transfrom(Train_Label)
        # thetha参数不为None时，利用thetha构造模型参数
        if theta is not None:
            self.Theta = theta
        else:
            # 随机生成服从标准正态分布的参数
            row = np.shape(self.Train_Data)[1]
            col = np.shape(self.Train_Label)[1]
            self.Theta = np.random.randn(row,col)

    def Transfrom(self,Train_Label):
        """
        这是将训练数据数字标签转换为one-hot标签
        :param Train_Label: 训练数据标签集
        """
        # 定义标签数组
        Label = []
        # 定义集合
        set = []
        # 首先遍历标签集，统计标签个数
        for label in Train_Label:
            if label not in set:
                set.append(label)
        # 遍历标签集，把每个标签转化为one-hot编码
        for label in Train_Label:
            _label = [0]*len(set)
            _label[label] = 1
            Label.append(_label)
        return np.array(Label)

    def Softmax(self,x):
        """
        这是Softmax函数的计算公式。
        :param x: 输入向量
        """
        # 获取向量中最大分量
        x_max = np.max(x)
        # 利用广播的形式对每个分量减去最大分量
        input = x - x_max
        # 计算向量指数运算后的和
        sum = np.sum(np.exp(input))
        # 计算Softmax函数值
        ans = np.exp(input) / sum
        return ans

    def Shuffle_Sequence(self):
        """
        这是在运行SGD算法或者MBGD算法之前，随机打乱后原始数据集的函数
        """
        # 首先获得训练集规模，之后按照规模生成自然数序列
        length = len(self.Train_Label)
        random_sequence = list(range(length))
        # 利用numpy的随机打乱函数打乱训练数据下标
        random_sequence = np.random.permutation(random_sequence)
        return random_sequence          # 返回数据集随机打乱后的数据序列

    def Gradient(self,train_data,train_label,predict_label):
        """
        这是计算Softmax回归模型参数的梯度函数
        :param train_data: 训练数据额
        :param train_label: 训练标签
        :param predict_label: 预测结果
        """
        error = (train_label - predict_label)
        thetha_gradient = train_data.T.dot(error)
        return thetha_gradient

    def BGD(self, alpha):
        """
        这是利用BGD算法进行一次迭代调整参数的函数
        :param alpha: 学习率
        """
        # 定义梯度增量数组
        gradient_increasment = []
        # 对输入的训练数据及其真实结果进行依次遍历
        for (train_data, train_label) in zip(self.Train_Data, self.Train_Label):
            # 首先计算train_data在当前模型预测结果
            train_data = np.reshape(train_data,(1,len(train_data)))
            train_label = np.reshape(train_label,(1,len(train_label)))
            predict = self.Softmax(train_data.dot(self.Theta))
            # 之后计算每组train_data的梯度增量，并放入梯度增量数组
            g = self.Gradient(train_data,train_label,predict)
            gradient_increasment.append(g)
        # 按列计算属性的平均梯度增量
        avg_g = np.average(gradient_increasment, 0)
        # 更新参数Theta
        self.Theta = self.Theta + alpha * avg_g

    def SGD(self, alpha):
        """
        这是利用SGD算法进行一次迭代调整参数的函数
        :param alpha: 学习率
        """
        # 首先将数据集随机打乱，减小数据集顺序对参数调优的影响
        shuffle_sequence = self.Shuffle_Sequence()
        # 对训练数据集进行遍历，利用每组训练数据对参数进行调整
        for index in shuffle_sequence:
            # 获取训练数据及其标签
            train_data = self.Train_Data[index]
            train_label = self.Train_Label[index]
            # 首先计算train_data在当前模型预测结果
            train_data = np.reshape(train_data,(1,len(train_data)))
            train_label = np.reshape(train_label, (1, len(train_label)))
            predict = self.Softmax(train_data.dot(self.Theta))[0]
            # 之后计算每组train_data的梯度增量
            g = self.Gradient(train_data, train_label, predict)
            # 更新模型参数Thetha
            self.Theta = self.Theta + alpha * g

    def MBGD(self, alpha, batch_size):
        """
        这是利用MBGD算法进行一次迭代调整参数的函数
        :param alpha: 学习率
        :param batch_size: 小批量样本规模
        """
        # 首先将数据集随机打乱，减小数据集顺序对参数调优的影响
        shuffle_sequence = self.Shuffle_Sequence()
        # 遍历每个小批量样本数据集及其标签
        for start in np.arange(0,len(shuffle_sequence),batch_size):
            # 判断start+batch_size是否大于数组长度，
            # 防止最后一组小样本规模可能小于batch_size的情况
            end = np.min([start+batch_size,len(shuffle_sequence)])
            # 获取训练小批量样本集及其标签
            mini_batch = shuffle_sequence[start:end]
            Mini_Train_Data = self.Train_Data[mini_batch]
            Mini_Train_Label = self.Train_Label[mini_batch]
            # 定义小批量训练数据集梯度增量数组
            gradient_increasment = []
            # 遍历每个小批量训练数据集
            for (train_data, train_label) in zip(Mini_Train_Data, Mini_Train_Label):
                # 首先计算train_data在当前模型预测结果
                train_data = np.reshape(train_data, (1, len(train_data)))
                train_label = np.reshape(train_label, (1, len(train_label)))
                predict = self.Softmax(train_data.dot(self.Theta))[0]
                # 之后计算每组train_data的梯度增量，并放入梯度增量数组
                g = self.Gradient(train_data, train_label, predict)
                gradient_increasment.append(g)
            # 按列计算属性的平均梯度增量
            avg_g = np.average(gradient_increasment, 0)
            # 更新参数Theta
            self.Theta = self.Theta + alpha * avg_g

    def Cost(self):
        """
        这是计算模型训练损失的函数
        """
        Cost = []
        for (train_data,train_label) in zip(self.Train_Data,self.Train_Label):
            # 首先计算train_data在当前模型预测结果
            train_data = np.reshape(train_data, (1, len(train_data)))
            predict = self.Softmax(train_data.dot(self.Theta))[0]
            # 加入1e-6是为了防止predict或者1-predict出现0的情况，从而防止对数运算出错
            cost = -train_label*np.log(predict+1e-6)
            Cost.append(np.sum(cost))
        return np.average(Cost)         # 返回每组训练数据训练损失之和

    def train_BGD(self, iter, alpha):
        """
        这是利用BGD算法迭代优化的函数
        :param iter: 迭代次数
        :param alpha: 学习率
        """
        # 定义训练损失数组，记录每轮迭代的训练数据集的训练损失
        Cost = []
        # 追加未开始训练的模型训练损失
        Cost.append(self.Cost())
        # 开始进行迭代训练
        for i in range(iter):
            # 利用学习率alpha，结合BGD算法对模型进行训练
            self.BGD(alpha)
            # 记录每次迭代的训练损失
            Cost.append(self.Cost())
        Cost = np.array(Cost)
        return Cost

    def train_SGD(self, iter, alpha):
        """
        这是利用SGD算法迭代优化的函数
        :param iter: 迭代次数
        :param alpha: 学习率
        """
        # 定义训练损失数组，记录每轮迭代的训练数据集的训练损失
        Cost = []
        # 追加未开始训练的模型训练损失
        Cost.append(self.Cost())
        # 开始进行迭代训练
        for i in range(iter):
            # 利用学习率alpha，结合SGD算法对模型进行训练
            self.SGD(alpha)
            # 记录每次迭代的训练损失
            Cost.append(self.Cost())
        Cost = np.array(Cost)
        return Cost

    def train_MBGD(self, iter, batch_size, alpha):
        """
        这是利用MBGD算法迭代优化的函数
        :param iter: 迭代次数
        :param batch_size: 小样本规模
        :param alpha: 学习率
        """
        # 定义训练损失数组，记录每轮迭代的训练数据集的训练损失
        Cost = []
        # 追加未开始训练的模型训练损失
        Cost.append(self.Cost())
        # 开始进行迭代训练
        for i in range(iter):
            # 利用学习率alpha，结合MBGD算法对模型进行训练
            self.MBGD(alpha, batch_size)
            # 记录每次迭代的训练损失
            Cost.append(self.Cost())
        Cost = np.array(Cost)
        return Cost

    def test(self, test_data):
        """
        这是对测试数据集的线性回归预测函数
        :param test_data: 测试数据集
        """
        # 定义预测结果数组
        predict_result = []
        # 对测试数据进行遍历，依次预测结果
        for data in test_data:
            # 预测每组data的结果
            predict_result.append(self.predict(data))
        predict_result = np.array(predict_result)
        return predict_result

    def predict(self, test_data):
        """
        这是对一组测试数据预测的函数
        :param test_data: 测试数据
        """
        # 对测试数据加入1维特征，以适应矩阵乘法
        tmp = [1.0]
        tmp.extend(test_data)
        # 计算test_data在当前模型预测结果
        test_data = np.array(tmp)
        # 首先计算train_data在当前模型预测结果
        test_data = np.reshape(test_data, (1, len(test_data)))
        predict = self.Softmax(test_data.dot(self.Theta))[0]
        #print(predict)
        return np.argmax(predict)       #返回最大概率对应的下标即分类
