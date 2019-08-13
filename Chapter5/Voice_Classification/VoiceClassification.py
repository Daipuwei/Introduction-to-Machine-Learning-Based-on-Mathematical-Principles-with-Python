#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 8:55
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : Voice_Classification.py
# @Software: PyCharm

from SoftmaxRegression.SoftmaxRegression import SoftmaxRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from Voice_Classification.Load_Voice_Data import Load_Voice_Data
import matplotlib as mpl
import matplotlib.pyplot as plt
import Other as Other
import numpy as np
import pandas as pd

def run_main():
    """
       这是主函数
    """
    # 导入语音信号数据集
    PATH = "./voice_data.txt"
    Data,Label = Load_Voice_Data(PATH)

    # 首先将数据随机打乱
    # 首先获得训练集规模，之后按照规模生成自然数序列
    length = len(Label)
    random_sequence = list(range(length))
    # 利用numpy的随机打乱函数打乱训练数据下标
    random_sequence = np.random.permutation(random_sequence)
    Data = Data[random_sequence]
    Label = Label[random_sequence]

    # 数据归一化
    #Data = MinMaxScaler().fit_transform(Data)

    # 把数据集分成训练数据集和测试数据集
    Train_Data,Test_Data,Train_Label,Test_Label = \
        train_test_split(Data,Label,test_size=0.2,random_state=10)
    Test_Label = Other.Transform(Test_Label)

    # 解决Matplotlib中的中文乱码问题，以便于后面实验结果可视化
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    #初始化模型参数
    theta = np.random.randn(np.shape(Train_Data)[1]+1,4)
    # 初始化相关Sofmax回归模型参数
    iteration = 20000                          # 迭代次数
    learning_rate = 0.01                       # 学习率
    batch_size = 32                            # 小批量样本规模
    lambds = [0,0.00001,0.0001,0.001,0.01]

    choices = ["c-","m-.","k--","g-.","r:"]
    col = ["民歌","古筝","摇滚","流行"]
    sheet_name = []         # 工作簿名称
    accuracy = []           # 精度数组
    precision = []          # 查准率
    recall = []             # 召回率
    f1 = []                 # f1度量
    # 首先构建一个Excel写入类的实例
    writer1 = pd.ExcelWriter("./不同L2正则化系数下的混淆矩阵.xlsx")
    writer2 = pd.ExcelWriter("./不同L2正则化系数下的训练误差统计信息.xlsx")
    # 遍历每个L2正则化系数，分别训练Softmax回归模型
    for lambd,choice in zip(lambds,choices):
        name = "lambda="+str(lambd)
        if lambd == 0:
            name = "无L2正则化"
        sheet_name.append(name)
        # 构造Softmax回归模型
        softmaxregression = SoftmaxRegression(Train_Data,Train_Label,theta)
        # 利用MBGD算法训练Softmax回归模型，加入L2正则化，
        # 当lambd为None时代表不加入L2正则化，并返训练迭代误差
        MBGD_Error = softmaxregression.train_MBGD(iteration,batch_size,learning_rate,lambd)
        # 可视化不同模型下的训练损失
        plt.plot(np.arange(len(MBGD_Error[500:])),MBGD_Error[500:],choice)
        # 计算不训练误差统计信息
        MBGD_Error = pd.DataFrame(MBGD_Error)
        print("训练误差的统计信息：")
        print(MBGD_Error.describe())
        MBGD_Error.describe().to_excel(writer2,sheet_name=name)
        # 利用Softmax回归模型对测试数据集进行预测
        Test_predict = softmaxregression.test(Test_Data)
        # 计算精度
        _accuracy = accuracy_score(Test_Label,Test_predict)
        # 计算查准率
        _precision = precision_score(Test_Label,Test_predict,average="micro")
        # 计算召回率
        _recall = recall_score(Test_Label,Test_predict,average="micro")
        # 计算f1
        _f1 = f1_score(Test_Label,Test_predict,average="micro")
        accuracy.append(_accuracy)
        precision.append(_precision)
        recall.append(_recall)
        f1.append(_f1)
        # 生成分类性能报告,由于classification_report函数结果
        # 为字符串，因此打印之后，手动保存到Excel
        report = classification_report(Test_Label,Test_predict,target_names=col)
        print("L2正则化系数为：%f的分类性能" % (lambd))
        print(report)
        print("精度为：",_accuracy)
        print("查准率为：", _precision)
        print("召回率为：", _recall)
        print("f1为：", _f1)
        # 计算混淆矩阵,并保存到excel
        sm_confusion_matrix = Other.confusion_matrix(Test_Label, Test_predict)
        sm_confusion_matrix = Other.Confusion_Matrix_Merge(sm_confusion_matrix, col)
        sm_confusion_matrix.to_excel(writer1,sheet_name=name)
    writer1.save()
    writer2.save()
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels=sheet_name,loc="best")
    plt.savefig("./训练损失.jpg",bbox_inches='tight')
    # 计算算法性能,并保存到excel
    Data = [accuracy, precision, recall, f1]
    Data = Other.Merge(Data,sheet_name,["精度","查准率","召回率","f1"])
    Data.to_excel("./不同L2正则化系数下的性能指标.xlsx")

if __name__ == '__main__':
    run_main()