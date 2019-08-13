#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 16:44
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : Voice_Classification.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from SoftmaxRegression.SoftmaxRegression import SoftmaxRegression
from .Load_Voice_Data import Load_Voice_Data
from Other import Merge
from Other import Confusion_Matrix_Merge
from Other import confusion_matrix
from Other import Transform

def run_main():
    """
       这是主函数
    """
    # 导入语音信号数据集
    PATH = "./voice_data.txt"
    Data,Label = Load_Voice_Data.Load_Voice_Data(PATH)

    # 数据归一化
    Data = MinMaxScaler().fit_transform(Data)

    # 分割训练集与测试集
    Train_Data, Test_Data, Train_Label, Test_Label = train_test_split \
        (Data, Label, test_size=0.25, random_state=10)
    Test_Label = Transform(Test_Label)

    # 初始化模型参数，参数维数比当前数据多一维
    # 是因为在Softmax回归模型内部会对数据扩展一维
    row,col = np.shape(Train_Data)[1]+1,4
    Theta = np.random.randn(row,col)

    # 解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 初始化相关Sofmax回归模型参数
    iteration = 100000                       # 迭代次数
    learning_rate = 0.1                      # 学习率
    batch_size = 256                         # 小批量样本规模

    # 寻求最佳的学习率
    learning_rates = [0.001,0.01,0.1,0.5]
    choices = ["b-","g-.",'r:','k--']
    col = ["民歌","古筝","摇滚","流行"]
    label= []               # 图例标签
    accuracy = []           # 精度数组
    # 混淆矩阵的行列名称
    name = list(np.arange(0,4))
    for i,num in enumerate(name):
        name[i] = str(num)
    # 首先构建一个Excel写入类的实例
    writer = pd.ExcelWriter("./不同学习率下的混淆矩阵.xlsx")
    # 遍历每个学习率，分别进行训练Softmax回归模型
    for learning_rate,choice in zip(learning_rates,choices):
        # 生成每组结果的图例标签
        _label = "learning_rate="+str(learning_rate)
        label.append(_label)
        # 构造MBGD优化算法的Softmax回归模型
        softmaxregression = SoftmaxRegression(Train_Data,Train_Label,Theta)
        # 利用MBGD算法训练Softmax回归模型，并返回迭代训练损失
        MBGD_Cost = softmaxregression.train_MBGD(iteration,batch_size,learning_rate)
        # 绘制迭代次数与迭代训练损失曲线
        plt.plot(np.arange(len(MBGD_Cost)),MBGD_Cost,choice)
        # 利用Logistic回归模型对测试数据集进行预测
        Test_predict = softmaxregression.test(Test_Data)
        # 计算精度
        accuracy.append(accuracy_score(Test_Label,Test_predict))
        # 生成分类性能报告,由于classification_report函数结果
        # 为字符串，因此打印之后，手动保存到Excel
        report = classification_report(Test_Label,Test_predict,target_names=col)
        print("学习率为：%f的分类性能"%(learning_rate))
        print(report)
        # 计算混淆矩阵,并保存到excel
        sm_confusion_matrix = confusion_matrix(Test_Label,Test_predict)
        sm_confusion_matrix = Confusion_Matrix_Merge(sm_confusion_matrix,col)
        sm_confusion_matrix.to_excel(writer,sheet_name=_label)
    writer.save()
    # 迭代训练损失可视化的相关操作
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels = label,loc = "best")
    plt.savefig("./不同学习率下MBGD的训练损失.jpg",bbox_inches='tight')
    #plt.show()
    plt.close()

    # 合并精度分类性能
    Data = [accuracy]
    col = ["精度"]
    Data = Merge(Data,label,col)
    # 将结果保存为Excel文档
    Data.to_excel("./不同学习率下MBGD的分类精度.xlsx")

    # 寻找最佳小样本规模
    batch_sizes = [64,128,256,512]
    choices = ["b-", "g-.", 'r:', 'k--']
    col = ["民歌", "古筝", "摇滚", "流行"]
    label = []              # 图例标签
    accuracy = []           # 精度数组
    Cost = []               # 训练损失
    # 遍历每个学习率，获取该学习率下的训练误差
    # 构造混淆矩阵的行列名称
    name = list(np.arange(0,4))
    for i,num in enumerate(name):
        name[i] = str(num)
    # 首先构建一个Excel写入类的实例
    writer = pd.ExcelWriter("./不同小批量样本下的混淆矩阵.xlsx")
    # 遍历每个batch_size，分别进行训练Softmax回归模型
    for batch_size, choice in zip(batch_sizes, choices):
        # 生成每组结果的图例标签
        _label = "batch_size=" + str(batch_size)
        label.append(_label)
        # 构造MBGD优化算法的Softmax回归模型
        softmaxregression = SoftmaxRegression(Train_Data, Train_Label,Theta)
        # 利用MBGD算法训练Softmax回归模型，并返回迭代训练损失
        MBGD_Cost = softmaxregression.train_MBGD(iteration, batch_size, learning_rate)
        Cost.append(MBGD_Cost)
        # 绘制迭代次数与迭代训练损失曲线
        plt.plot(np.arange(len(MBGD_Cost)), MBGD_Cost, choice)
        # 利用Logistic回归模型对测试数据集进行预测
        Test_predict = softmaxregression.test(Test_Data)
        # 计算精度
        accuracy.append(accuracy_score(Test_Label,Test_predict))
        # 生成分类性能报告,由于classification_report函数结果
        # 为字符串，因此打印之后，手动保存到Excel
        report = classification_report(Test_Label,Test_predict,target_names=col)
        print("小批量样本为：%d的分类性能"%(batch_size))
        print(report)
        # 计算混淆矩阵,并保存到excel
        sm_confusion_matrix = confusion_matrix(Test_Label, Test_predict)
        sm_confusion_matrix = Confusion_Matrix_Merge(sm_confusion_matrix, col)
        sm_confusion_matrix.to_excel(writer, sheet_name=_label)
    writer.save()
    # 迭代训练损失可视化的相关操作
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels=label, loc="best")
    plt.savefig("./不同小批量样本规模下MBGD的训练损失.jpg", bbox_inches='tight')
    #plt.show()
    plt.close()

    # 合并分类分类性能
    Data = [accuracy]
    col = ["精度"]
    Data = Merge(Data, label, col)
    # 将结果保存为Excel文档
    Data.to_excel("./不同小批量样本规模下MBGD的精度.xlsx")

    for cost, choice in zip(Cost, choices):
        # 绘制迭代次数与迭代训练误差曲线
        plt.plot(np.arange(len(cost[200:])), cost[200:], choice)
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.xlim((200,100000))
    plt.grid(True)
    plt.legend(labels=label, loc="best")
    plt.savefig("./不同小批量样本规模下MBGD的训练损失部分结果.jpg", bbox_inches='tight')
    #plt.show()
    plt.close()

if __name__ == '__main__':
    run_main()