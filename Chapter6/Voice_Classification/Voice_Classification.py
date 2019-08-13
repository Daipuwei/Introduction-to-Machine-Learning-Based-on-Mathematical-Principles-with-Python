#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/24 15:32
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-506实验室
# @File    : Voice_Classification.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from BPNN.BPNN import BPNN
from sklearn.preprocessing import StandardScaler
from Other import Transform
from Other import Merge
import pandas as pd

def Load_Voice_Data(path, one_hot=False):
    """
    这是导入数据的函数
    :param path: 数据文件的路径
    :return: 数据集
    """
    Data = []
    Label = []
    with open(path) as f:
        for line in f.readlines():
            str = line.strip().split("\t")
            data = []
            for i in range(1, len(str)):
                data.append(float(str[i]))
            Data.append(data)
            Label.append(int(str[0]))
    if one_hot == True:
        for (index, label) in enumerate(Label):
            _label = [0] * 4
            _label[label - 1] = 1
            Label[index] = _label
    Data = np.array(Data)
    Label = np.array(Label)
    return Data, Label


def run_main():
    """
       这是主函数
    """
    # 导入语音数据集数据集
    path = "./voice_data.txt"
    Data, Label = Load_Voice_Data(path, True)

    # 数据归一化
    Data = StandardScaler().fit_transform(Data)

    # 初始化BPNN模型参数
    INPUT_NODE = np.shape(Data)[1]  # 输入层神经元个数
    OUTPUT_NODE = 4  # 输出层神经元个数
    HIDDEN_NODE = int(round((INPUT_NODE * OUTPUT_NODE) * 2.0 / 3))  # 隐含层神经元个数

    # 解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 寻求最佳的学习率
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    choices = ["b-", "g-.", 'r:', 'k--']
    label = []  # 图例标签
    Accuracy = []  # 精度数组
    F1 = []  # F1系数
    Precision = []  # 查准率
    Recall = []  # 召回率
    Train_Loss = []  # 训练损失
    Test_Loss = []  # 测试损失
    Train_Accuracy = []  # 训练精度
    Test_Accuracy = []  # 测试精度
    # 初始化相关BPNN模型参数
    iteration = 2  # 迭代次数
    batch_size = 16  # 小批量样本规模
    beta = 0  # 动量因子
    lambd = 0.0001  # L2正则化系数
    for learning_rate, choice in zip(learning_rates, choices):
        # 生成每组结果的图例标签
        _label = "learning_rate=" + str(learning_rate)
        label.append(_label)
        # 生成交叉验证的训练集与测试集的下标组合
        kf = KFold(n_splits=10, shuffle=True, random_state=np.random.randint(0, len(Data)))
        train_test_index = []
        for (train_index, test_index) in kf.split(Data):
            train_test_index.append((train_index, test_index))
        accuracy = []  # 精度数组
        f1 = []  # F1系数
        precision = []  # 查准率
        recall = []  # 召回率
        train_loss = []  # 训练损失
        test_loss = []  # 测试损失
        train_accuracy = []  # 训练精度
        test_accuracy = []  # 测试精度
        for (train_index, test_index) in train_test_index:
            # 初始化权重和阈值
            input_hidden_weights = np.random.randn(INPUT_NODE, HIDDEN_NODE)
            hidden_threshold = np.random.randn(1, HIDDEN_NODE)
            hidden_output_weights = np.random.randn(HIDDEN_NODE, OUTPUT_NODE)
            output_threshold = np.random.randn(1, OUTPUT_NODE)
            # 构造MBGD优化算法的BPNN模型
            BPNN_MBGD = BPNN(INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE,
                             input_hidden_weights, hidden_threshold,
                             hidden_output_weights, output_threshold)
            # 将数据集分成训练数据集与测试数据集
            Train_Data = Data[train_index]
            Train_Label = Label[train_index]
            Test_Data = Data[test_index]
            Test_Label = Label[test_index]
            '''print(np.shape(Train_Data))
            print(np.shape(Train_Label))
            print(np.shape(Test_Data))
            print(np.shape(Test_Label))'''
            # 利用MBGD算法训练BPNN，并返训练迭代误差
            _train_loss, _test_loss, _train_accuracy, _test_accuracy = \
                BPNN_MBGD.train_SGD(Train_Data, Train_Label, Test_Data, Test_Label,
                                 iteration, learning_rate, beta, lambd)
            '''BPNN_MBGD.train_MBGD(Train_Data, Train_Label, Test_Data, Test_Label,
                                                 iteration, batch_size, learning_rate, beta, lambd)'''
            train_loss.append(_train_loss)
            test_loss.append(_test_loss)
            train_accuracy.append(_train_accuracy)
            test_accuracy.append(_test_accuracy)
            # 利用BPNN模型对测试数据集进行预测
            Test_predict = BPNN_MBGD.test(Test_Data, one_hot=False)
            # 计算分类性能
            accuracy.append(accuracy_score(Transform(Test_Label), Test_predict))
            f1.append(f1_score(Transform(Test_Label), Test_predict, average="micro"))
            precision.append(precision_score(Transform(Test_Label), Test_predict, average="micro"))
            recall.append(recall_score(Transform(Test_Label), Test_predict, average="micro"))
        # 计算交叉验证平均训练损失
        Train_Loss.append(np.average(train_loss, 0))
        Test_Loss.append(np.average(test_loss, 0))
        Train_Accuracy.append(np.average(train_accuracy, 0))
        Test_Accuracy.append(np.average(test_accuracy, 0))
        # 计算交叉验证平均性能
        Accuracy.append(np.average(accuracy))  # 计算精度
        F1.append(np.average(f1))  # 计算f1
        Precision.append(np.average(precision))  # 计算查准率
        Recall.append(np.average(recall))  # 计算召回率
        # 绘制迭代次数与迭代训练误差曲线
        plt.plot(np.arange(len(train_loss[0])), np.average(train_loss, 0), choice)
    # 训练迭代损失可视化的相关操作
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels=label, loc="best")
    plt.savefig("./不同学习率下MBGD的训练损失.jpg", bbox_inches='tight')
    # plt.show()
    plt.close()

    data = pd.DataFrame(np.array(Train_Loss).T, columns=label)
    data.to_excel("./不同学习率下MBGD的训练损失.xlsx")
    data.describe().to_excel("./不同学习率下MBGD的训练损失数据分析.xlsx")

    data = pd.DataFrame(np.array(Test_Loss).T, columns=label)
    data.to_excel("./不同学习率下MBGD的测试损失.xlsx")
    data.describe().to_excel("./不同学习率下MBGD的测试损失数据分析.xlsx")

    data = pd.DataFrame(np.array(Train_Accuracy).T, columns=label)
    data.to_excel("./不同学习率下MBGD的训练精度.xlsx")
    data.describe().to_excel("./不同学习率下MBGD的训练精度数据分析.xlsx")

    data = pd.DataFrame(np.array(Test_Accuracy).T, columns=label)
    data.to_excel("./不同学习率下MBGD的测试精度.xlsx")
    data.describe().to_excel("./不同学习率下MBGD的测试精度数据分析.xlsx")

    # 合并分类分类性能
    data = [Accuracy]
    col = ["精度"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同学习率下MBGD的精度.xlsx")
    # plt.show()
    plt.close()

    # 合并分类分类性能
    data = [F1]
    col = ["f1"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同学习率下MBGD的f1.xlsx")

    # 合并分类分类性能
    data = [Precision]
    col = ["查准率"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同学习率下MBGD的查准率.xlsx")

    # 合并分类分类性能
    data = [Recall]
    col = ["召回率"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同学习率下MBGD的召回率.xlsx")

    # 寻找最佳小样本规模
    batch_sizes = [16, 32, 64, 128]
    choices = ["b-", "g-.", 'r:', 'k--']
    label = []  # 图例标签
    Accuracy = []  # 精度数组
    F1 = []  # F1系数
    Precision = []  # 查准率
    Recall = []  # 召回率
    Train_Loss = []  # 训练损失
    Test_Loss = []  # 测试损失
    Train_Accuracy = []  # 训练精度
    Test_Accuracy = []  # 测试精度
    # 初始化相关BPNN模型参数
    iteration = 2  # 迭代次数
    learning_rate = 0.001  # 学习率
    beta = 0  # 动量因子
    lambd = 0.0001  # L2正则化系数
    for batch_size, choice in zip(batch_sizes, choices):
        # 生成每组结果的图例标签
        _label = "batch_size=" + str(batch_size)
        label.append(_label)
        # 生成交叉验证的训练集与测试集的下标组合
        kf = KFold(n_splits=10, shuffle=True, random_state=np.random.randint(0, len(Data)))
        train_test_index = []
        for (train_index, test_index) in kf.split(Data):
            train_test_index.append((train_index, test_index))
        accuracy = []  # 精度数组
        f1 = []  # F1系数
        precision = []  # 查准率
        recall = []  # 召回率
        train_loss = []  # 训练损失
        test_loss = []  # 测试损失
        train_accuracy = []  # 训练精度
        test_accuracy = []  # 测试精度
        for (train_index, test_index) in train_test_index:
            # 初始化权重和阈值
            input_hidden_weights = np.random.randn(INPUT_NODE, HIDDEN_NODE)
            hidden_threshold = np.random.randn(1, HIDDEN_NODE)
            hidden_output_weights = np.random.randn(HIDDEN_NODE, OUTPUT_NODE)
            output_threshold = np.random.randn(1, OUTPUT_NODE)
            # 构造MBGD优化算法的BPNN模型
            BPNN_MBGD = BPNN(INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE,
                             input_hidden_weights, hidden_threshold,
                             hidden_output_weights, output_threshold)
            # 将数据集分成训练数据集与测试数据集
            Train_Data = Data[train_index]
            Train_Label = Label[train_index]
            Test_Data = Data[test_index]
            Test_Label = Label[test_index]
            # 利用MBGD算法训练BPNN，并返训练迭代误差
            _train_loss, _test_loss, _train_accuracy, _test_accuracy = \
                BPNN_MBGD.train_BGD(Train_Data, Train_Label, Test_Data, Test_Label,
                                    iteration, learning_rate, beta, lambd)
            '''BPNN_MBGD.train_MBGD(Train_Data, Train_Label, Test_Data, Test_Label,
                                                 iteration, batch_size, learning_rate, beta, lambd)'''
            train_loss.append(_train_loss)
            test_loss.append(_test_loss)
            train_accuracy.append(_train_accuracy)
            test_accuracy.append(_test_accuracy)
            # 利用BPNN模型对测试数据集进行预测
            Test_predict = BPNN_MBGD.test(Test_Data, one_hot=False)
            # 计算分类性能
            accuracy.append(accuracy_score(Transform(Test_Label), Test_predict))
            f1.append(f1_score(Transform(Test_Label), Test_predict, average="micro"))
            precision.append(precision_score(Transform(Test_Label), Test_predict, average="micro"))
            recall.append(recall_score(Transform(Test_Label), Test_predict, average="micro"))
        # 计算交叉验证平均训练损失
        Train_Loss.append(np.average(train_loss, 0))
        Test_Loss.append(np.average(test_loss, 0))
        Train_Accuracy.append(np.average(train_accuracy, 0))
        Test_Accuracy.append(np.average(test_accuracy, 0))
        # 计算交叉验证平均性能
        Accuracy.append(np.average(accuracy))  # 计算精度
        F1.append(np.average(accuracy))  # 计算f1
        Precision.append(np.average(accuracy))  # 计算查准率
        Recall.append(np.average(accuracy))  # 计算召回率
        # 绘制迭代次数与迭代训练误差曲线
        plt.plot(np.arange(len(train_loss[0])), np.average(train_loss, 0), choice)
    # 训练迭代损失可视化的相关操作
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels=label, loc="best")
    plt.savefig("./不同小批量样本规模下MBGD的训练损失.jpg", bbox_inches='tight')
    # plt.show()
    plt.close()

    data = pd.DataFrame(np.array(Train_Loss).T, columns=label)
    data.to_excel("./不同小批量样本规模下MBGD的训练损失.xlsx")
    data.describe().to_excel("./不同小批量样本规模下MBGD的训练损失数据分析.xlsx")

    data = pd.DataFrame(np.array(Test_Loss).T, columns=label)
    data.to_excel("./不同小批量样本规模下MBGD的测试损失.xlsx")
    data.describe().to_excel("./不同小批量样本规模下MBGD的测试损失数据分析.xlsx")

    data = pd.DataFrame(np.array(Train_Accuracy).T, columns=label)
    data.to_excel("./不同小批量样本规模下MBGD的训练精度.xlsx")
    data.describe().to_excel("./不同小批量样本规模下MBGD的训练精度数据分析.xlsx")

    data = pd.DataFrame(np.array(Test_Accuracy).T, columns=label)
    data.to_excel("./不同小批量样本规模下MBGD的测试精度.xlsx")
    data.describe().to_excel("./不同小批量样本规模下MBGD的测试精度数据分析.xlsx")

    # 合并分类分类性能
    data = [Accuracy]
    col = ["精度"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同小批量样本规模下MBGD的精度.xlsx")

    # 合并分类分类性能
    data = [F1]
    col = ["f1"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同小批量样本规模下MBGD的f1.xlsx")

    # 合并分类分类性能
    data = [Precision]
    col = ["查准率"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同小批量样本规模下MBGD的查准率.xlsx")

    # 合并分类分类性能
    data = [Recall]
    col = ["召回率"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同小批量样本规模下MBGD的召回率.xlsx")

    # 寻找最佳正则化系数
    lambds = [0.00001, 0.0001, 0.001, 0.01]
    choices = ["b-", "g-.", 'r:', 'k--']
    label = []  # 图例标签
    Accuracy = []  # 精度数组
    F1 = []  # F1系数
    Precision = []  # 查准率
    Recall = []  # 召回率
    Train_Loss = []  # 训练损失
    Test_Loss = []  # 测试损失
    Train_Accuracy = []  # 训练精度
    Test_Accuracy = []  # 测试精度
    # 初始化相关BPNN模型参数
    iteration = 20000  # 迭代次数
    learning_rate = 0.001  # 学习率
    batch_size = 64  # 小批量样本规模
    beta = 0  # 动量因子
    for lambd, choice in zip(lambds, choices):
        # 生成每组结果的图例标签
        _label = "lambd=" + str(lambd)
        label.append(_label)
        # 生成交叉验证的训练集与测试集的下标组合
        kf = KFold(n_splits=10, shuffle=True, random_state=np.random.randint(0, len(Data)))
        train_test_index = []
        for (train_index, test_index) in kf.split(Data):
            train_test_index.append((train_index, test_index))
        accuracy = []  # 精度数组
        f1 = []  # F1系数
        precision = []  # 查准率
        recall = []  # 召回率
        train_loss = []  # 训练损失
        test_loss = []  # 测试损失
        train_accuracy = []  # 训练精度
        test_accuracy = []  # 测试精度
        for (train_index, test_index) in train_test_index:
            # 初始化权重和阈值
            input_hidden_weights = np.random.randn(INPUT_NODE, HIDDEN_NODE)
            hidden_threshold = np.random.randn(1, HIDDEN_NODE)
            hidden_output_weights = np.random.randn(HIDDEN_NODE, OUTPUT_NODE)
            output_threshold = np.random.randn(1, OUTPUT_NODE)
            # 构造MBGD优化算法的BPNN模型
            BPNN_MBGD = BPNN(INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE,
                             input_hidden_weights, hidden_threshold,
                             hidden_output_weights, output_threshold)
            # 将数据集分成训练数据集与测试数据集
            Train_Data = Data[train_index]
            Train_Label = Label[train_index]
            Test_Data = Data[test_index]
            Test_Label = Label[test_index]
            # 利用MBGD算法训练BPNN，并返训练迭代误差
            _train_loss, _test_loss, _train_accuracy, _test_accuracy = \
                BPNN_MBGD.train_BGD(Train_Data, Train_Label, Test_Data, Test_Label,
                                    iteration, learning_rate, beta, lambd)
            '''BPNN_MBGD.train_MBGD(Train_Data, Train_Label, Test_Data, Test_Label,
                                                 iteration, batch_size, learning_rate, beta, lambd)'''
            train_loss.append(_train_loss)
            test_loss.append(_test_loss)
            train_accuracy.append(_train_accuracy)
            test_accuracy.append(_test_accuracy)
            # 利用BPNN模型对测试数据集进行预测
            Test_predict = BPNN_MBGD.test(Test_Data, one_hot=False)
            # 计算分类性能
            accuracy.append(accuracy_score(Transform(Test_Label), Test_predict))
            f1.append(f1_score(Transform(Test_Label), Test_predict, average="micro"))
            precision.append(precision_score(Transform(Test_Label), Test_predict, average="micro"))
            recall.append(recall_score(Transform(Test_Label), Test_predict, average="micro"))
        # 计算交叉验证平均训练损失
        Train_Loss.append(np.average(train_loss, 0))
        Test_Loss.append(np.average(test_loss, 0))
        Train_Accuracy.append(np.average(train_accuracy, 0))
        Test_Accuracy.append(np.average(test_accuracy, 0))
        # 计算交叉验证平均性能
        Accuracy.append(np.average(accuracy))  # 计算精度
        F1.append(np.average(accuracy))  # 计算f1
        Precision.append(np.average(accuracy))  # 计算查准率
        Recall.append(np.average(accuracy))  # 计算召回率
        # 绘制迭代次数与迭代训练误差曲线
        plt.plot(np.arange(len(train_loss[0])), np.average(train_loss, 0), choice)
    # 训练迭代损失可视化的相关操作
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels=label, loc="best")
    plt.savefig("./不同L2正则化系数下MBGD的训练损失.jpg", bbox_inches='tight')
    # plt.show()
    plt.close()

    data = pd.DataFrame(np.array(Train_Loss).T, columns=label)
    data.to_excel("./不同L2正则化系数下MBGD的训练损失.xlsx")
    data.describe().to_excel("./不同L2正则化系数下MBGD的训练损失数据分析.xlsx")

    data = pd.DataFrame(np.array(Test_Loss).T, columns=label)
    data.to_excel("./不同L2正则化系数下MBGD的测试损失.xlsx")
    data.describe().to_excel("./不同L2正则化系数下MBGD的测试损失数据分析.xlsx")

    data = pd.DataFrame(np.array(Train_Accuracy).T, columns=label)
    data.to_excel("./不同L2正则化系数下MBGD的训练精度.xlsx")
    data.describe().to_excel("./不同L2正则化系数下MBGD的训练精度数据分析.xlsx")

    data = pd.DataFrame(np.array(Test_Accuracy).T, columns=label)
    data.to_excel("./不同L2正则化系数下MBGD的测试精度.xlsx")
    data.describe().to_excel("./不同L2正则化系数下MBGD的测试精度数据分析.xlsx")

    # 合并分类分类性能
    data = [Accuracy]
    col = ["精度"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同L2正则化系数下MBGD的精度.xlsx")

    # 合并分类分类性能
    data = [F1]
    col = ["f1"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同L2正则化系数下MBGD的f1.xlsx")

    # 合并分类分类性能
    data = [Precision]
    col = ["查准率"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同L2正则化系数下MBGD的查准率.xlsx")

    # 合并分类分类性能
    data = [Recall]
    col = ["召回率"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同L2正则化系数下MBGD的召回率.xlsx")

    # 寻找最佳动量因子
    betas = [0, 0.5, 0.9, 0.99]  # 动量因子数组
    choices = ["b-", "g-.", 'r:', 'k--']
    label = []  # 图例标签
    Accuracy = []  # 精度数组
    F1 = []  # F1系数
    Precision = []  # 查准率
    Recall = []  # 召回率
    Train_Loss = []  # 训练损失
    Test_Loss = []  # 测试损失
    Train_Accuracy = []  # 训练精度
    Test_Accuracy = []  # 测试精度
    # 初始化相关BPNN模型参数
    iteration = 2  # 迭代次数
    learning_rate = 0.001  # 学习率
    batch_size = 64  # 小批量样本规模
    lambd = 0.0001  # L1正则化系数
    for beta, choice in zip(betas, choices):
        # 生成每组结果的图例标签
        _label = "beta=" + str(beta)
        label.append(_label)
        # 生成交叉验证的训练集与测试集的下标组合
        kf = KFold(n_splits=10, shuffle=True, random_state=np.random.randint(0, len(Data)))
        train_test_index = []
        for (train_index, test_index) in kf.split(Data):
            train_test_index.append((train_index, test_index))
        accuracy = []  # 精度数组
        f1 = []  # F1系数
        precision = []  # 查准率
        recall = []  # 召回率
        train_loss = []  # 训练损失
        test_loss = []  # 测试损失
        train_accuracy = []  # 训练精度
        test_accuracy = []  # 测试精度
        for (train_index, test_index) in train_test_index:
            # 初始化权重和阈值
            input_hidden_weights = np.random.randn(INPUT_NODE, HIDDEN_NODE)
            hidden_threshold = np.random.randn(1, HIDDEN_NODE)
            hidden_output_weights = np.random.randn(HIDDEN_NODE, OUTPUT_NODE)
            output_threshold = np.random.randn(1, OUTPUT_NODE)
            # 构造MBGD优化算法的BPNN模型
            BPNN_MBGD = BPNN(INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE,
                             input_hidden_weights, hidden_threshold,
                             hidden_output_weights, output_threshold)
            # 将数据集分成训练数据集与测试数据集
            Train_Data = Data[train_index]
            Train_Label = Label[train_index]
            Test_Data = Data[test_index]
            Test_Label = Label[test_index]
            # 利用MBGD算法训练BPNN，并返训练迭代误差
            _train_loss, _test_loss, _train_accuracy, _test_accuracy = \
                BPNN_MBGD.train_BGD(Train_Data, Train_Label, Test_Data, Test_Label,
                                    iteration, learning_rate, beta, lambd)
            '''BPNN_MBGD.train_MBGD(Train_Data, Train_Label, Test_Data, Test_Label,
                                                 iteration, batch_size, learning_rate, beta, lambd)'''
            train_loss.append(_train_loss)
            test_loss.append(_test_loss)
            train_accuracy.append(_train_accuracy)
            test_accuracy.append(_test_accuracy)
            # 利用BPNN模型对测试数据集进行预测
            Test_predict = BPNN_MBGD.test(Test_Data, one_hot=False)
            # 计算分类性能
            accuracy.append(accuracy_score(Transform(Test_Label), Test_predict))
            f1.append(f1_score(Transform(Test_Label), Test_predict, average="micro"))
            precision.append(precision_score(Transform(Test_Label), Test_predict, average="micro"))
            recall.append(recall_score(Transform(Test_Label), Test_predict, average="micro"))
        # 计算交叉验证平均训练损失
        Train_Loss.append(np.average(train_loss, 0))
        Test_Loss.append(np.average(test_loss, 0))
        Train_Accuracy.append(np.average(train_accuracy, 0))
        Test_Accuracy.append(np.average(test_accuracy, 0))
        # 计算交叉验证平均性能
        Accuracy.append(np.average(accuracy))  # 计算精度
        F1.append(np.average(accuracy))  # 计算f1
        Precision.append(np.average(accuracy))  # 计算查准率
        Recall.append(np.average(accuracy))  # 计算召回率
        # 绘制迭代次数与迭代训练误差曲线
        plt.plot(np.arange(len(train_loss[0])), np.average(train_loss, 0), choice)
    # 训练迭代损失可视化的相关操作
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels=label, loc="best")
    plt.savefig("./不同L2正则化系数下MBGD的训练损失.jpg", bbox_inches='tight')
    # plt.show()
    plt.close()

    data = pd.DataFrame(np.array(Train_Loss).T, columns=label)
    data.to_excel("./不同动量因子下MBGD的训练损失.xlsx")
    data.describe().to_excel("./不同动量因子下MBGD的训练损失数据分析.xlsx")

    data = pd.DataFrame(np.array(Test_Loss).T, columns=label)
    data.to_excel("./不同动量因子下MBGD的测试损失.xlsx")
    data.describe().to_excel("./不同动量因子下MBGD的测试损失数据分析.xlsx")

    data = pd.DataFrame(np.array(Train_Accuracy).T, columns=label)
    data.to_excel("./不同动量因子下MBGD的训练精度.xlsx")
    data.describe().to_excel("./不同动量因子下MBGD的训练精度数据分析.xlsx")

    data = pd.DataFrame(np.array(Test_Accuracy).T, columns=label)
    data.to_excel("./不同动量因子下MBGD的测试精度.xlsx")
    data.describe().to_excel("./不同动量因子下MBGD的测试精度数据分析.xlsx")

    # 合并分类分类性能
    data = [Accuracy]
    col = ["精度"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同动量因子下MBGD的精度.xlsx")

    # 合并分类分类性能
    data = [F1]
    col = ["f1"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同动量因子下MBGD的f1.xlsx")

    # 合并分类分类性能
    data = [Precision]
    col = ["查准率"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同动量因子下MBGD的查准率.xlsx")

    # 合并分类分类性能
    data = [Recall]
    col = ["召回率"]
    data = Merge(data, label, col)
    # 将结果保存为Excel文档
    data.to_excel("./不同动量因子下MBGD的召回率.xlsx")


if __name__ == '__main__':
    run_main()