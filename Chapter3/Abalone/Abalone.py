#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/18 20:22
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : Abalone.py
# @Software: PyCharm

from LocalWeightedLinearRegression.LocalWeightedLinearRegression \
    import LocalWeightedLinearRegression as LWLR
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def Load_Abalone(path):
    """
    这是导入鲍鱼数据集的函数
    :param path: 文件路径
    """
    # 定义鲍鱼数据和结果数组
    data = []
    result = []
    # 打开路径为path的文件
    with open(path) as f:
        # 遍历文件中的每一行
        for line in f.readlines():
            str = line.strip().split(',')
            tmp = []
            length = len(str[1:])
            # 鲍鱼数据集中的第一个属性是性别，该属性属于离散数据
            # 因此导入数据时必须抛开这一列，最后一列是环数，加1.5可以预测年龄
            for (index,s) in enumerate(str[1:]):
                # 最后一个数据追加到result
                if index == length-1:
                    result.append(float(s)+1.5)
                # 否则剩下的数据追加到tmp临时数组
                else:
                    tmp.append(float(s))
            # 一组数据追加到数据集中
            data.append(tmp)
        data = np.array(data)
        result = np.array(result)
    return data,result

def Merge(data,col):
    """
    这是生成DataFrame数据的函数
    :param data:输入数据
    :param col:列名称数组
    """
    Data = np.array(data).T
    return pd.DataFrame(Data,columns=col)

def run_main():
    """
       这是主函数
    """
    # 导入鲍鱼数据集
    path = "./abalone_data.txt"
    Data,Result = Load_Abalone(path)

    # 把数据集分成训练集合测试集
    Train_Data,Test_Data,Train_Result,Test_Result = train_test_split\
        (Data,Result,test_size=0.01,random_state=10)

    # 解决Matplotlib中的中文乱码问题，以便于后面实验结果可视化
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 可视化测试集
    col = ['长度','直径','高度','总重量','剥壳重量','内脏重量','壳重']

    # 遍历局部加权线性回归算法的预测结果
    fig = plt.figure()
    for (index, c) in enumerate(col):
        if index == 0:
            ax = fig.add_subplot(311)
        else:
            ax = fig.add_subplot(334+index-1)
        ax.scatter(Test_Data[:,index], Test_Result, alpha=0.5, c='b', s=10)
        ax.grid(True)
        plt.xlabel(c)
        plt.ylabel("鲍鱼年龄")
        # 子图之间使用紧致布局
        plt.tight_layout()
    plt.savefig("./测试结果可视化.jpg", bbox_inches='tight')
    plt.close()

    for (index,c) in enumerate(col):
        plt.scatter(Test_Data[:,index],Test_Result,alpha=0.5,c='b',s=10)
        plt.grid(True)
        plt.xlabel(c)
        plt.ylabel("鲍鱼年龄")
        plt.savefig("./"+c+"可视化.jpg",bbox_inches='tight')
        #plt.show()
        plt.close()

    # 初始化局部加权线性回归模型
    lwlr = LWLR(Train_Data,Train_Result)

    # 初始化局部加权回归带宽系数参数
    K = [0.0001,0.001,0.003,0.005,0.01,0.05,0.1,0.3,0.5]
    tmp = list(np.arange(1,101))
    K.extend(tmp)
    predict_result = []
    Loss = []
    # 把测试集进行从小到大排序
    sort = np.argsort(Test_Data[:, 1])
    Test_Data = Test_Data[sort]
    Test_Result = Test_Result[sort]
    # 遍历每个带宽系数，利用局部加权线性回归算法进行预测,并计算预测误差
    for k in K:
        # 在带宽系数k下，利用局部加权线性回归进行预测
        predict = lwlr.predict_NormalEquation(Test_Data,k)
        #print(np.shape(predict))
        # 计算每组数据的预测误差
        loss = (Test_Result-predict)**2
        print("k=%f时的误差：%f"%(k,np.sum(loss)))
        predict_result.append(predict)
        Loss.append(loss)
        # 可视化预测结果
        plt.scatter(Test_Data[:, 1], Test_Result, alpha=0.5, c='b', s=10)
        plt.plot(Test_Data[:,1],predict,'r')
        plt.grid(True)
        plt.xlabel('直径')
        plt.ylabel("鲍鱼年龄")
        plt.savefig("./k=" + str(k) + "可视化.jpg", bbox_inches='tight')
        plt.close()

    # 部分预测结果可视化
    # k = [0.1,0.3,1,3,10,100]
    k = [0.1,0.3,1,3,10,100]
    index= [6,7,9,11,18,108]
    # 遍历每个带宽系数，利用局部加权线性回归算法进行预测,并计算预测误差
    fig = plt.figure()
    for (j,(i,k_)) in enumerate(zip(index,k)):
        # 在带宽系数k下，利用局部加权线性回归进行预测
        predict = predict_result[i]
        # 可视化预测结果
        ax = fig.add_subplot(230+j+1)
        ax.scatter(Test_Data[:, 1], Test_Result, alpha=0.5, c='b', s=10)
        ax.plot(Test_Data[:,1],predict,'r')
        ax.grid(True)
        ax.legend(labels=["k=" + str(k_)], loc="best")
        plt.xlabel('直径')
        plt.ylabel("鲍鱼年龄")
        plt.tight_layout()
    plt.savefig("./部分预测结果.jpg")

    # 保存预测数据
    data = [Test_Result]
    col = ["真实结果"]
    for (index,k) in enumerate(K):
        data.append(predict_result[index])
        col.append("k="+str(k))
    Data = Merge(data,col)
    Data.to_excel("./真实结果与预测结果.xlsx")

    # 保存预测误差结果及其统计信息
    data = []
    col = []
    for (index, k) in enumerate(K):
        data.append(Loss[index])
        col.append("k=" + str(k))
    Data = Merge(data,col)
    Data.to_excel("./预测误差.xlsx")
    information = Data.describe()
    information.to_excel("./预测误差统计信息.xlsx")

    # 可视化不同带宽参数的局部加权线性回归模型在测试集的均方误差和预测标准差
    K = list(np.arange(1, 101))
    col = ["LWLR-MSE", "LWLR-std"]
    LWLR_MSE = list(information.loc['mean'])[9:]
    LWLR_std = list(information.loc['std'])[9:]
    plt.plot(K, LWLR_MSE, 'b')
    plt.plot(K, LWLR_std, 'c-.')
    plt.grid(True)
    plt.legend(labels=col, loc='best')
    plt.xlabel("带宽系数")
    plt.savefig("./局部加权线性回归的预测均方误差和标准差.jpg", bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    run_main()