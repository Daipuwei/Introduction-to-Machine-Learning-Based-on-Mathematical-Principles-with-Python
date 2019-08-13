#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/9 14:54
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-506实验室
# @File    : BPNN.py
# @Software: PyCharm

import numpy as np
from copy import deepcopy
from Activation_Function.Activation_Function import Sigmoid
from Activation_Function.Activation_Function import Sigmoid_Derivative
from sklearn.metrics import accuracy_score
from Other import Transform

"""
    这是经典3层BP神经网络类代码。输入层与隐含层、隐含层与输出层的激活函数
    都是使用Sigmoid函数。使用BP算法与梯度下降算法进行优化参数。在利用梯度
    下降算法优化参数时，本模型实现了BGD、SGD和MBGD三种梯度下降算法,并且实
    现了动量梯度优化算法，输入层与隐含层、隐含层与输出层之间的权重和隐含层、
    输出层的阈值都采用随机生成，并且服从标准正态分布。损失函数采用的预测结
    果与训练标签之间的均方误差。
"""

class BPNN(object):
    def __init__(self, input_n,hidden_n,output_n,
                      input_hidden_weights = None,hidden_threshold = None,
                      hidden_output_weights = None,output_threshold = None):
        """
        这是BP神经网络类的构造函数
        :param input_n: 输入层神经元个数
        :param hidden_n: 隐藏层神经元个数
        :param output_n: 输出层神经元个数
        :param input_hidden_weights: 输入层与隐含层之间的权重
        :param hidden_threshold: 隐含层的阈值
        :param hidden_output_weights: 隐含层与输出层之间的权重
        :param output_threshold: 输出层的阈值
        """
        # 初始化输入层、隐含层和输出层的神经元个数
        self.input_n = input_n                # 输入层神经元个数
        self.hidden_n = hidden_n              # 隐含层神经元个数
        self.output_n = output_n              # 输出层神经元个数
        # 初始化输入层、隐含层和输出层的神经元
        # 输入层神经元
        self.input_cells = np.zeros(self.input_n).reshape((1,self.input_n))
        # 隐含层神经元
        self.hidden_cells = np.zeros(self.hidden_n).reshape((1,self.hidden_n))
        # 输出层神经元
        self.output_cells = np.zeros(self.output_n).reshape((1,self.output_n))
        # 初始化输入层与隐含层之间的权重
        # 输入层与隐含层之间的权重为None，则随机生成
        if input_hidden_weights is None:
            self.input_hidden_weights = np.random.randn(self.input_n,self.hidden_n)
        else: # 输入层与隐含层之间的权重不为None，则直接导入
            # 使用深拷贝则避免数据更改造成不必要的麻烦，若直接
            # 复制属于引用，在迭代过程中self.input_hidden_weights
            # 进行更新，也会导致input_hidden_weights更新，
            # input_hidden_weights的更新会占用cpu内存，
            # 导致BP神经网络的训练缓慢。
            self.input_hidden_weights = deepcopy(input_hidden_weights)
        # 初始化隐含层的阈值
        if hidden_threshold is None:
            # 隐含层的阈值为None，则随机生成
            self.hidden_threshold = np.random.randn(1, self.hidden_n)
        else:# 输入层与隐含层之间的权重不为None，则直接导入
            # 使用深拷贝则避免数据更改造成不必要的麻烦，若直接
            # 复制属于引用，在迭代过程中self.hidden_threshold
            # 进行更新，也会导致hidden_threshold更新，
            # hidden_threshold的更新会占用cpu内存，
            # 导致BP神经网络的训练缓慢。
            self.hidden_threshold = deepcopy(hidden_threshold)
        # 初始化隐含层与输出层之间的权重
        if hidden_output_weights is None:
            # 隐含层与输出层之间的权重为None，则随机生成
            self.hidden_output_weights = np.random.randn(self.hidden_n,self.output_n)
        else:# 输入层与隐含层之间的权重不为None，则直接导入
            # 使用深拷贝则避免数据更改造成不必要的麻烦，若直接复制属于引用，
            # 在迭代过程中self.hidden_output_weights进行更新，
            # 也会导致hidden_output_weights更新input_hidden_weights的更新
            # 会占用cpu内存，导致BP神经网络的训练缓慢。
            self.hidden_output_weights = deepcopy(hidden_output_weights)
        # 初始化输出层的阈值
        if output_threshold is None:
            # 输出层的阈值为None，则随机生成
            self.output_threshold = np.random.randn(1, self.output_n)
        else:# 输入层与隐含层之间的权重不为None，则直接导入
            # 使用深拷贝则避免数据更改造成不必要的麻烦，若直接
            # 复制属于引用，在迭代过程中self.output_threshold进行
            # 更新，也会导致output_threshold更新，output_threshold的
            # 更新会占用cpu内存，导致BP神经网络的训练缓慢。
            self.output_threshold = deepcopy(output_threshold)
        # 初始化超参数的动量
        # 输入层与隐含层之间权重的动量
        self.input_hidden_weights_momentum = 0
        # 隐含层阈值的动量
        self.hidden_threshold_momentum = 0
        # 隐含层与输出层之间的权重的动量
        self.hidden_output_weights_momentum = 0
        # 输出层阈值的动量
        self.output_threshold_momentum = 0

    def predict(self,input):
        """
        这是BP神经网络向前学习传递的函数
        :param input: 输入数据
        :return: 返回输出层预测结果
        """
        # 初始化输入神经元
        self.input_cells = deepcopy(input)
        # 输入层到隐含层的前向传播过程，计算隐含层输出
        self.hidden_cells = self.input_cells.dot(self.input_hidden_weights)
        self.hidden_cells = Sigmoid(self.hidden_cells + self.hidden_threshold)
        # 隐含层到输出层的前向传播过程，计算输出层输出
        self.output_cells = self.hidden_cells.dot(self.hidden_output_weights)
        self.output_cells = Sigmoid(self.output_cells + self.output_threshold)
        return self.output_cells

    def Gradient(self,train_label,predict_label):
        """
        这是计算在反馈过程中相关参数的梯度增量的函数,
        损失函数为训练标签与预测标签之间的均方误差
        :param train_label: 训练数据标签
        :param predict_label: BP神经网络的预测结果
        """
        # 这是输出层与隐含层之间的反向传播过程，计算隐
        # 含层与输出层之间的权重与输出层的阈值的梯度增量
        error = predict_label - train_label
        derivative = Sigmoid_Derivative(predict_label)
        g = derivative * error
        #print("g的形状：",np.shape(g))
        # 计算隐藏层与输出层之间权重的梯度增量
        hidden_output_weights_gradient_increasement = self.hidden_cells.T.dot(g)
        # 计算隐藏层阈值的梯度增量
        output_threshold_gradient_increasement = g
        # 这是隐含层与输入层之间的反向传播过程，计算输
        # 入层与隐含层之间的权重与隐含层的阈值的梯度增量
        e = Sigmoid_Derivative(self.hidden_cells) * g.dot(self.hidden_output_weights.T)
        #print("e的形状：",np.shape(e))
        # 计算输入层与隐藏层之间权重的梯度增量
        input_hidden_weights_gradient_increasement = self.input_cells.T.dot(e)
        # 计算输入层阈值的梯度增量
        hidden_threshold_gradient_increasement = e
        return hidden_output_weights_gradient_increasement,\
               output_threshold_gradient_increasement,\
               input_hidden_weights_gradient_increasement,\
               hidden_threshold_gradient_increasement

    def back_propagate(self,hidden_output_weights_gradient_increasement,
                       output_threshold_gradient_increasement,
                       input_hidden_weights_gradient_increasement,
                       hidden_threshold_gradient_increasement,
                       learning_rate,beta,lambd):
        """
        这是利用误差反向传播算法对BP神经网络的模型参数进行迭代更新的函数
        :param hidden_output_weights_gradient_increasement:隐含层与输出层之间的权重的梯度增量
        :param output_threshold_gradient_increasement:输出层的阈值的梯度增量
        :param input_hidden_weights_gradient_increasement:输入层与隐含层之间的权重的梯度增量
        :param hidden_threshold_gradient_increasement:隐含层的阈值的梯度增量
        :param learning_rate: 学习率
        :param beta: 指数衰减系数（动量因子），默认为0
        :param lambd: L2正则化系数，默认为0
        """
        # 更新各个超参数的动量
        self.input_hidden_weights_momentum = \
            beta * self.input_hidden_weights_momentum + (1 - beta) * \
            input_hidden_weights_gradient_increasement
        self.hidden_threshold_momentum = \
            beta * self.hidden_threshold_momentum + (1 - beta) * \
            hidden_threshold_gradient_increasement
        self.hidden_output_weights_momentum = \
            beta * self.hidden_output_weights_momentum + (1 - beta) * \
            hidden_output_weights_gradient_increasement
        self.output_threshold_momentum = \
            beta * self.output_threshold_momentum + (1 - beta) * \
            output_threshold_gradient_increasement
        # 隐含层与输出层之间的权重的更新
        self.hidden_output_weights = \
            (1 - lambd * learning_rate) * self.hidden_output_weights - \
            learning_rate * self.hidden_output_weights_momentum
        #输出层的阈值更新
        self.output_threshold = self.output_threshold - learning_rate * \
                                 self.output_threshold_momentum
        #输入层与隐含层之间的权重更新
        self.input_hidden_weights = \
            (1 - lambd * learning_rate) * self.input_hidden_weights - \
            learning_rate * self.input_hidden_weights_momentum
        #隐含层的阈值更新
        self.hidden_threshold = self.hidden_threshold - learning_rate * \
                                self.hidden_threshold_momentum

    def MSE(self,Train_Label,Predict_Label):
        """
        这是计算训练集的训练损失的函数
        :param Train_Label: 训练标签集
        :param Predict_Label: 预测结果集
        """
        # 计算训练数据集的均方误差
        mse = np.average((Train_Label-Predict_Label)**2)/2
        return mse

    def BGD(self,learning_rate,beta,lambd):
        """
        这是利用BGD算法进行一次迭代调整参数的函数
        :param learning_rate: 学习率
        :param beta: 指数衰减系数（动量因子），默认为0
        :param lambd: L2正则化系数，默认为0
        """
        # 分别定义BP神经网络模型参数梯度增量数组
        # 隐含层与输出层之间的权重的梯度增量
        hidden_output_weights_gradient_increasements = []
        # 输出层的阈值的梯度增量
        output_threshold_gradient_increasements = []
        # 输入层与隐含层之间的权重的梯度增量
        input_hidden_weights_gradient_increasements = []
        # 隐含层的阈值的梯度增量
        hidden_threshold_gradient_increasements = []
        # 定义训练数据集预测结果数组
        Predict_Label = []
        # 遍历整个训练数据集
        for (train_data,train_label) in zip(self.Train_Data,self.Train_Label):
            # 首先计算train_data在当前模型预测结果
            train_data = np.reshape(train_data, (1, len(train_data)))
            train_label = np.reshape(train_label, (1, len(train_label)))
            # 对训练数据train_data进行预测，并保存预测结果
            predict = self.predict(train_data)
            Predict_Label.append(predict)
            # 计算BP神经网络在每组train_data下的各个模型参数
            # 的梯度增量，并放入梯度增量数组
            hidden_output_weights_gradient_increasement\
                , output_threshold_gradient_increasement\
                , input_hidden_weights_gradient_increasement\
                , hidden_threshold_gradient_increasement \
                = self.Gradient(train_label,predict)
            hidden_output_weights_gradient_increasements.append\
                (hidden_output_weights_gradient_increasement)
            output_threshold_gradient_increasements.append\
                (output_threshold_gradient_increasement)
            input_hidden_weights_gradient_increasements.append\
                (input_hidden_weights_gradient_increasement)
            hidden_threshold_gradient_increasements.append\
                (hidden_threshold_gradient_increasement)
        # 对参数的梯度增量求取平均值
        hidden_output_weights_gradient_increasement_avg = \
            np.average(hidden_output_weights_gradient_increasements, 0)
        output_threshold_gradient_increasement_avg = \
            np.average(output_threshold_gradient_increasements, 0)
        input_hidden_weights_gradient_increasement_avg = \
            np.average(input_hidden_weights_gradient_increasements, 0)
        hidden_threshold_gradient_increasement_avg = \
            np.average(hidden_threshold_gradient_increasements, 0)
        # 对BP神经网络模型参数进行更新
        self.back_propagate(hidden_output_weights_gradient_increasement_avg,
                            output_threshold_gradient_increasement_avg,
                            input_hidden_weights_gradient_increasement_avg,
                            hidden_threshold_gradient_increasement_avg,
                            learning_rate,beta,lambd)
        # 计算训练数据集的训练损失
        mse = self.MSE(self.Train_Label, Predict_Label)
        return mse

    def SGD(self,learning_rate,beta,lambd):
        """
        这是利用SGD算法进行一次迭代调整参数的函数
        :param learning_rate: 学习率
        :param beta: 指数衰减系数（动量因子），默认为0
        :param lambd: L2正则化系数c，默认为0
        """
        # 获取随机序列
        random_sequence = self.Shuffle_Sequence(self.Size)
        # 随机打乱整个训练数据集及其标签
        self.Train_Data = self.Train_Data[random_sequence]
        self.Train_Label = self.Train_Label[random_sequence]
        # 定义训练数据集预测结果数组
        Predict_Label = []
        # 遍历整个训练数据集
        for (train_data,train_label) in zip(self.Train_Data,self.Train_Label):
            # 首先计算train_data在当前模型预测结果
            train_data = np.reshape(train_data, (1, len(train_data)))
            train_label = np.reshape(train_label, (1, len(train_label)))
            # 对训练数据train_data进行预测，并保存预测结果
            predict = self.predict(train_data)
            Predict_Label.append(predict)
            # 计算BP神经网络在每组train_data下的各个模型参数的
            # 梯度增量，并放入梯度增量数组
            hidden_output_weights_gradient_increasement\
                ,output_threshold_gradient_increasement\
                , input_hidden_weights_gradient_increasement\
                , hidden_threshold_gradient_increasement \
                = self.Gradient(train_label,predict)
            # 对BP神经网络模型参数进行更新
            self.back_propagate(hidden_output_weights_gradient_increasement/self.Size,
                                output_threshold_gradient_increasement/self.Size,
                                input_hidden_weights_gradient_increasement/self.Size,
                                hidden_threshold_gradient_increasement/self.Size,
                                learning_rate,beta,lambd)
        # 计算训练数据集的训练损失
        mse = self.MSE(self.Train_Label, Predict_Label)
        return mse

    def MBGD(self,batch_size,learning_rate,beta,lambd):
        """
        这是利用MBGD算法进行一次迭代调整参数的函数
        :param batch_size: 小批量样本规模
        :param learning_rate: 学习率
        :param decay_rate: 学习率衰减指数
        :param decay_step: 学习率衰减步数
        :param beta: 指数衰减系数（动量因子），默认为0
        :param lambd: L2正则化系数，默认为0
        """
        # 获取随机序列
        random_sequence = self.Shuffle_Sequence(self.Size)
        # 随机打乱整个训练数据集及其标签
        self.Train_Data = self.Train_Data[random_sequence]
        self.Train_Label = self.Train_Label[random_sequence]
        # 定义小批量数据集上均方误差数组
        MSE = []
        # 更新当前迭代次数与学习率
        # 遍历每个小批量样本数据集及其标签
        for start in np.arange(0,self.Size, batch_size):
            # 判断start+batch_size是否大于数组长度，
            # 防止最后一组小样本规模可能小于batch_size的情况
            end = np.min([start + batch_size, self.Size])
            # 获取训练小批量样本集及其标签
            Mini_Train_Data = self.Train_Data[start:end]
            Mini_Train_Label = self.Train_Label[start:end]
            # 在小批量样本上利用BGD算法对模型参数
            # 进行更新,并计算训练数据集的均方误差
            # 首先分别定义BP神经网络模型参数梯度增量数组
            # 隐含层与输出层之间的权重的梯度增量
            hidden_output_weights_gradient_increasements = []
            # 输出层的阈值的梯度增量
            output_threshold_gradient_increasements = []
            # 输入层与隐含层之间的权重的梯度增量
            input_hidden_weights_gradient_increasements = []
            # 隐含层的阈值的梯度增量
            hidden_threshold_gradient_increasements = []
            #print(self.learning_rate_decay)
            # 定义小批量训练集的预测结果数组
            Predict_Label = []
            # 遍历整个小批量训练数据集
            for (train_data, train_label) in zip(Mini_Train_Data, Mini_Train_Label):
                # 首先计算train_data在当前模型预测结果
                train_data = np.reshape(train_data, (1, len(train_data)))
                train_label = np.reshape(train_label, (1, len(train_label)))
                # 对训练数据train_data进行预测，并保存预测结果
                predict = self.predict(train_data)
                Predict_Label.append(predict)
                # 计算BP神经网络在每组train_data下的各个模型参数
                # 的梯度增量，并放入梯度增量数组
                hidden_output_weights_gradient_increasement\
                    , output_threshold_gradient_increasement\
                    , input_hidden_weights_gradient_increasement\
                    , hidden_threshold_gradient_increasement \
                    = self.Gradient(train_label, predict)
                hidden_output_weights_gradient_increasements.\
                    append(hidden_output_weights_gradient_increasement)
                output_threshold_gradient_increasements.\
                    append(output_threshold_gradient_increasement)
                input_hidden_weights_gradient_increasements.\
                    append(input_hidden_weights_gradient_increasement)
                hidden_threshold_gradient_increasements.\
                    append(hidden_threshold_gradient_increasement)
            # 对参数的梯度增量求取平均值
            hidden_output_weights_gradient_increasement_avg = \
                np.average(hidden_output_weights_gradient_increasements,0)
            output_threshold_gradient_increasement_avg = \
                np.average(output_threshold_gradient_increasements, 0)
            input_hidden_weights_gradient_increasement_avg = \
                np.average(input_hidden_weights_gradient_increasements, 0)
            hidden_threshold_gradient_increasement_avg = \
                np.average(hidden_threshold_gradient_increasements, 0)
            # 对BP神经网络模型参数进行更新
            self.back_propagate(hidden_output_weights_gradient_increasement_avg,
                                output_threshold_gradient_increasement_avg,
                                input_hidden_weights_gradient_increasement_avg,
                                hidden_threshold_gradient_increasement_avg,
                                learning_rate,beta,lambd)
            # 计算小批量训练数据集的训练损失
            mse = self.MSE(Mini_Train_Label, Predict_Label)
            MSE.append(mse*len(Mini_Train_Label))
        # 计算整个训练集上的训练损失
        MSE = np.sum(MSE)/self.Size
        return MSE

    def Shuffle_Sequence(self,size):
        """
        这是在运行SGD算法或者MBGD算法之前，随机打乱后原始数据集的函数
        :param size: 数据集规模
        """
        # 首先按照数据集规模生成自然数序列
        random_sequence = list(range(size))
        # 利用numpy的随机打乱函数打乱训练数据下标
        random_sequence = np.random.permutation(random_sequence)
        return random_sequence          # 返回数据集随机打乱后的数据序列

    def Init(self,Train_Data,Train_Label):
        """
        这是初始化训练数据集的函数
        :param Train_Data: 训练数据集
        :param Train_Label: 训练标签集
        :return:
        """
        self.Train_Data = Train_Data            # 训练数据集
        self.Train_Label = Train_Label          # 训练数据集标签
        self.Size = np.shape(Train_Data)[0]     # 训练数据集规模性

    def train_BGD(self,Train_Data,Train_Label,Test_Data,Test_Label,
                  iteration,learning_rate,beta,lambd):
        """
        这是利用BGD算法对训练数据集进行迭代训练的函数
        :param Train_Data: 训练数据集
        :param Train_Label: 训练标签集
        :param Test_Data: 测试数据集
        :param Test_Label: 测试标签集
        :param iteration: 迭代次数
        :param learning_rate: 学习率
        :param beta: 指数衰减系数（动量因子），默认为0
        :param lambd: L2正则化系数，默认为0
        """
        # 初始化训练数据集及其标签
        self.Init(Train_Data,Train_Label)
        # 定义每次迭代过程的训练损失和测试损失的数组
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        # 首先不利拥动量梯度下降算法进行一次更新，保存初始动量方向
        self.BGD(learning_rate, 0, lambd)
        # 进行iteration次迭代训练
        for i in range(iteration):
            # 利用BGD算法对模型参数进行更新,并获取训练集的训练损失
            _train_loss = self.BGD(learning_rate, beta, lambd)
            _train_loss += 0.5 * lambd * learning_rate * \
                           (np.sum(self.input_hidden_weights ** 2)
                            + np.sum(self.hidden_output_weights ** 2))
            train_loss.append(_train_loss)
            Train_Predict_Label = self.test(Train_Data, True)
            train_accuracy.append(accuracy_score(Transform(Train_Label)
                                   , Transform(Train_Predict_Label)))
            # 对测试数据集进行预测,并计算精度
            Test_predict = self.test(Test_Data, one_hot=False)
            test_accuracy.append(accuracy_score(Transform(Test_Label), Test_predict))
            # 遍历测试数据集，进行预测计算测试损失
            Test_Predict_Label = []
            for test_data in Test_Data:
                Test_Predict_Label.append(self.predict(test_data))
            test_loss.append(self.MSE(Test_Label, Test_Predict_Label))
        return np.array(train_loss), np.array(test_loss), \
               np.array(train_accuracy), np.array(test_accuracy)

    def train_SGD(self,Train_Data,Train_Label,Test_Data,Test_Label,
                  iteration,learning_rate,beta,lambd):
        """
        这是利用SGD算法对训练数据集进行迭代训练的函数
        :param Train_Data: 训练数据集
        :param Train_Label: 训练标签集
        :param Test_Data: 测试数据集
        :param Test_Label: 测试标签集
        :param iteration: 迭代次数
        :param learning_rate: 学习率
        :param beta: 指数衰减系数（动量因子），默认为0
        :param lambd: L2正则化系数，默认为0
        """
        # 初始化训练数据集及其标签
        self.Init(Train_Data,Train_Label)
        # 定义每次迭代过程的训练损失和测试损失的数组
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        # 首先不利拥动量梯度下降算法进行一次更新，保存初始动量方向
        self.SGD(learning_rate, 0, lambd)
        # 进行iteration次迭代训练
        for i in range(iteration):
            # 利用SGD算法对模型参数进行更新,并获取训练集的训练损失
            _train_loss = self.SGD(learning_rate,beta,lambd)
            _train_loss += 0.5 * lambd * learning_rate * \
                           (np.sum(self.input_hidden_weights ** 2)
                            + np.sum(self.hidden_output_weights ** 2))
            train_loss.append(_train_loss)
            Train_Predict_Label = self.test(Train_Data, True)
            train_accuracy.append(accuracy_score(Transform(Train_Label)
                                  , Transform(Train_Predict_Label)))
            # 对测试数据集进行预测,并计算精度
            Test_predict = self.test(Test_Data, one_hot=False)
            test_accuracy.append(accuracy_score(Transform(Test_Label), Test_predict))
            # 遍历测试数据集，进行预测计算测试损失
            Test_Predict_Label = []
            for test_data in Test_Data:
                Test_Predict_Label.append(self.predict(test_data))
            test_loss.append(self.MSE(Test_Label, Test_Predict_Label))
        return np.array(train_loss), np.array(test_loss), \
               np.array(train_accuracy), np.array(test_accuracy)

    def train_MBGD(self,Train_Data,Train_Label,Test_Data,Test_Label,
                   iteration,batch_size,learning_rate,beta,lambd):
        """
        这是利用BGD算法对训练数据集进行迭代训练的函数
        :param Train_Data: 训练数据集
        :param Train_Label: 训练标签集
        :param Test_Data: 测试数据集
        :param Test_Label: 测试标签集
        :param iteration: 迭代次数
        :param batch_size: 小样本规模
        :param learning_rate: 学习率
        :param beta: 指数衰减系数（动量因子），默认为0
        :param lambd: L2正则化系数，默认为0
        """
        # 初始化训练数据集及其标签
        self.Init(Train_Data,Train_Label)
        # 定义每次迭代过程的训练损失和测试损失的数组
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        # 首先不利拥动量梯度下降算法进行一次更新，保存初始动量方向
        self.MBGD(batch_size, learning_rate, 0, lambd)
        # 进行iteration次迭代训练
        for i in range(iteration):
            # 利用MBGD算法对模型参数进行更新,并计算训练集的训练损失
            _train_loss = self.MBGD(batch_size,learning_rate,beta,lambd)
            _train_loss += 0.5*lambd*learning_rate*\
                           (np.sum(self.input_hidden_weights**2)
                             + np.sum(self.hidden_output_weights**2))
            train_loss.append(_train_loss)
            Train_Predict_Label = self.test(Train_Data,True)
            train_accuracy.append(accuracy_score(Transform(Train_Label)
                                  ,Transform(Train_Predict_Label)))
            # 对测试数据集进行预测,并计算精度
            Test_predict = self.test(Test_Data, one_hot=False)
            test_accuracy.append(accuracy_score(Transform(Test_Label), Test_predict))
            # 遍历测试数据集，进行预测计算测试损失
            Test_Predict_Label = []
            for test_data in Test_Data:
                Test_Predict_Label.append(self.predict(test_data)[0])
            test_loss.append(self.MSE(Test_Label, Test_Predict_Label))
        return np.array(train_loss),np.array(test_loss),\
               np.array(train_accuracy),np.array(test_accuracy)

    def test(self,Test_Data,one_hot = False):
        """
        这是BP神经网络测试函数
        :param Test_Data: 测试数据集
        """
        predict_labels = []
        for test_data in Test_Data:
            # 对测试数据进行预测
            test_data = np.reshape(test_data,(1,len(test_data)))
            predict_output = self.predict(test_data)
            # 计算预测分类
            index = np.argmax(predict_output)
            if one_hot == True:
                # 生成标准输出神经元并置0
                tmp = [0] * self.output_n
                tmp[index] = 1
                predict_labels.append(tmp)
            else:
                predict_labels.append(index)
        predict_labels = np.array(predict_labels)
        return predict_labels