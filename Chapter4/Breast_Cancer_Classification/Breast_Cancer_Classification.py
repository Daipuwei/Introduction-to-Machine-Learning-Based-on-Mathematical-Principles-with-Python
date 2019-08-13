#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 13:03
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : Breast_Cancer_Classification.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from LogisticRegression.LogisticRegression import LogisticRegression
from Other import Merge
from Other import Confusion_Matrix_Merge

def run_main():
    """
       这是主函数
    """
    # 导入乳腺癌数据集
    breast_cancer = load_breast_cancer()
    Data,Label = breast_cancer.data,breast_cancer.target
    print(Data)

    # 数据归一化
    Data = MinMaxScaler().fit_transform(Data)

    #分割训练集与测试集
    Train_Data,Test_Data,Train_Label,Test_Label = train_test_split\
        (Data,Label,test_size=0.25,random_state=10)

    # 初始化模型参数，参数维数比当前数据多一维
    # 是因为在Logistic回归模型内部会对数据扩展一维
    size = np.shape(Train_Data)[1]+1
    Thetha = np.random.randn(size)

    # 构建不同梯度下降算法的Logistic回归模型
    LR_BGD = LogisticRegression(Train_Data,Train_Label,Thetha)              # BGD的Logistic回归模型
    LR_SGD = LogisticRegression(Train_Data, Train_Label, Thetha)            # SGD的Logistic回归模型
    LR_MBGD = LogisticRegression(Train_Data, Train_Label, Thetha)           # MBGD的Logistic回归模型

    # 初始化相关Logistic回归模型参数
    iteration = 20000                   # 迭代次数
    learning_rate = 0.1                 # 学习率
    batch_size = 32                     # 小批量样本规模

    # 训练Logistic回归模型
    BGD_Cost = LR_BGD.train_BGD(iteration, learning_rate)
    SGD_Cost = LR_SGD.train_SGD(iteration, learning_rate)
    MBGD_Cost = LR_MBGD.train_MBGD(iteration, batch_size, learning_rate)

    # 解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 三种模型在训练阶段的损失变化
    col = ['BGD','SGD','MBGD']
    x = np.arange(len(BGD_Cost))
    plt.plot(x,BGD_Cost,'r')
    plt.plot(x,SGD_Cost,'b--')
    plt.plot(x,MBGD_Cost,'k-.')
    plt.grid(True)
    plt.legend(labels=col,loc='best')
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.savefig("./三种算法的训练损失.jpg",bbox_inches='tight')
    #plt.show()
    plt.close()

    # 利用Logistic回归预测是否患乳腺癌
    BGD_predict = LR_BGD.predict(Test_Data)
    SGD_predict = LR_SGD.predict(Test_Data)
    MBGD_predict = LR_MBGD.predict(Test_Data)

    # 计算预测精度
    BGD_accuracy = accuracy_score(Test_Label,BGD_predict)
    SGD_accuracy = accuracy_score(Test_Label,SGD_predict)
    MBGD_accuracy = accuracy_score(Test_Label,MBGD_predict)
    accuracy = [BGD_accuracy,SGD_accuracy,MBGD_accuracy]
    print("BGD算法的精度：", BGD_accuracy)
    print("SGD算法的精度：", SGD_accuracy)
    print("MBGD算法的精度：", MBGD_accuracy)

    # 计算查准率
    BGD_precision = precision_score(Test_Label, BGD_predict)
    SGD_precision = precision_score(Test_Label, SGD_predict)
    MBGD_precision = precision_score(Test_Label, MBGD_predict)
    precision = [BGD_precision, SGD_precision, MBGD_precision]
    print("BGD算法的查准率：", BGD_precision)
    print("SGD算法的查准率：", SGD_precision)
    print("MBGD算法的查准率：", MBGD_precision)

    # 计算召回率
    BGD_recall = recall_score(Test_Label, BGD_predict)
    SGD_recall = recall_score(Test_Label, SGD_predict)
    MBGD_recall = recall_score(Test_Label, MBGD_predict)
    recall = [BGD_recall, SGD_recall, MBGD_recall]
    print("BGD算法的召回率：", BGD_recall)
    print("SGD算法的召回率：", SGD_recall)
    print("MBGD算法的召回率：", MBGD_recall)

    # 计算f1度量
    BGD_f1 = f1_score(Test_Label,BGD_predict)
    SGD_f1 = f1_score(Test_Label,SGD_predict)
    MBGD_f1 = f1_score(Test_Label,MBGD_predict)
    f1 = [BGD_f1, SGD_f1, MBGD_f1]
    print("BGD算法的f1：", BGD_f1)
    print("SGD算法的f1：", SGD_f1)
    print("MBGD算法的f1：", MBGD_f1)

    # 计算混淆矩阵
    BGD_confusion_matrix = confusion_matrix(Test_Label, BGD_predict)
    SGD_confusion_matrix = confusion_matrix(Test_Label, SGD_predict)
    MBGD_confusion_matrix = confusion_matrix(Test_Label, MBGD_predict)
    print("BGD算法的混淆矩阵：", BGD_confusion_matrix)
    print("SGD算法的混淆矩阵：", SGD_confusion_matrix)
    print("MBGD算法的混淆矩阵：", MBGD_confusion_matrix)

    # 合并分类性能指标
    col = ["精度","查准率","召回率","f1"]
    row = ["BGD","SGD","MBGD"]
    Result = [accuracy,precision,recall,f1]
    Result = Merge(Result,row,col)
    # 将结果保存为Excel文档
    Result.to_excel("./三种算法的分类性能指标.xlsx")

    # 保存混淆矩阵
    # 首先构建一个Excel写入类的实例
    writer = pd.ExcelWriter("./混淆矩阵.xlsx")
    # 之后把BGD、SGD和MBGD算法下的混淆矩阵转化为DataFrame
    BGD_confusion_matrix = Confusion_Matrix_Merge(BGD_confusion_matrix,["正例","反例"])
    SGD_confusion_matrix = Confusion_Matrix_Merge(SGD_confusion_matrix,["正例","反例"])
    MBGD_confusion_matrix = Confusion_Matrix_Merge(MBGD_confusion_matrix,["正例","反例"])
    # 依次把DateFrame以sheet的形式写入Excel
    BGD_confusion_matrix.to_excel(writer,sheet_name="BGD")
    SGD_confusion_matrix.to_excel(writer, sheet_name="SGD")
    MBGD_confusion_matrix.to_excel(writer, sheet_name="MBGD")
    writer.save()

    # 寻求最佳的学习率
    learning_rates = [0.001,0.01,0.05,0.1,0.3]
    choices = ["b-","g-.",'r:','k--','y-']
    label= []           # 图例标签
    accuracy = []       # 精度数组
    precision = []      # 查准率数组
    recall = []         # 召回率数组
    f1 = []             # f1度量数组
    # 遍历每个学习率，获取该学习率下Logistic回归模型的训练误差
    for learning_rate,choice in zip(learning_rates,choices):
        # 生成每组结果的图例标签
        label.append("learning_rate="+str(learning_rate))
        # 构造MBGD优化算法的Logistic回归模型
        LR_MBGD = LogisticRegression(Train_Data,Train_Label,Thetha)
        # 利用MBGD算法训练Logistic回归模型，并返训练迭代误差
        MBGD_Cost = LR_MBGD.train_MBGD(iteration,batch_size,learning_rate)
        # 绘制迭代次数与迭代训练误差曲线
        plt.plot(np.arange(len(MBGD_Cost)),MBGD_Cost,choice)
        # 利用Logistic回归模型对测试数据集进行预测
        Test_predict = LR_MBGD.predict(Test_Data)
        # 计算精度、查准率、召回率和f1分类性能指标
        accuracy.append(accuracy_score(Test_Label,Test_predict))
        f1.append(f1_score(Test_Label,Test_predict))
        precision.append((precision_score(Test_Label,Test_predict)))
        recall.append(recall_score(Test_Label,Test_predict))
    # 合并分类性能指标结果
    Data = [accuracy,precision,recall,f1]
    Data = Merge(Data,label,col)
    # 将结果保存为Excel文档
    Data.to_excel("./不同学习率下MBGD的评价指标.xlsx")
    # 训练迭代损失可视化的相关操作
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels = label,loc = "best")
    plt.savefig("./不同学习率下MBGD的训练损失.jpg",bbox_inches='tight')
    plt.show()
    plt.close()

    # 寻找最佳小样本规模
    batch_sizes = [16,32,64,128,256]
    choices = ["b-", "g-.", 'r:', 'k--', 'y-']
    label= []           # 图例标签
    accuracy = []       # 精度数组
    precision = []      # 查准率数组
    recall = []         # 召回率数组
    f1 = []             # f1度量数组
    # 遍历每个学习率，获取该学习率下的训练损失
    for batch_size, choice in zip(batch_sizes, choices):
        label.append("batch_size="+str(batch_size))
        # 构造MBGD优化算法的Logistic回归模型
        LR_MBGD = LogisticRegression(Train_Data, Train_Label, Thetha)
        # 利用MBGD算法训练Logistic回归模型，并返迭代训练损失
        MBGD_Cost = LR_MBGD.train_MBGD(iteration, batch_size, learning_rate)
        # 绘制迭代次数与迭代训练损失曲线
        plt.plot(np.arange(len(MBGD_Cost)), MBGD_Cost, choice)
        # 利用Logistic回归模型对测试数据集进行预测
        Test_predict = LR_MBGD.predict(Test_Data)
        # 计算精度、查准率、召回率和f1分类性能指标
        accuracy.append(accuracy_score(Test_Label, Test_predict))
        f1.append(f1_score(Test_Label, Test_predict))
        precision.append((precision_score(Test_Label, Test_predict)))
        recall.append(recall_score(Test_Label, Test_predict))
    # 合并分类性能指标结果
    Data = [accuracy, precision, recall, f1]
    Data = Merge(Data, label, col)
    # 将结果保存为Excel文档
    Data.to_excel("./不同小批量样本规模下MBGD的评价指标.xlsx")
    # 迭代训练损失可视化的相关操作
    plt.xlabel("迭代次数")
    plt.ylabel("训练损失")
    plt.grid(True)
    plt.legend(labels=label, loc="best")
    plt.savefig("./不同小批量样本规模下MBGD的训练损失.jpg", bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    run_main()