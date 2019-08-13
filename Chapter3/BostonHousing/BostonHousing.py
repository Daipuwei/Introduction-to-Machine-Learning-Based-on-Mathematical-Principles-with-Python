#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/1316:45
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : BostonHousing.py
# @Software: PyCharm

from LinearRegression.LinearRegression import LinearRegression
from LocalWeightedLinearRegression.LocalWeightedLinearRegression \
    import LocalWeightedLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    # 导入数据以及划分训练数据与测试数据
    InputData, Result = load_boston(return_X_y=True)
    # 为了方便实验，只取第6维特征。第6列为平均房间数目
    InputData = np.array(InputData)[:,5]
    # 保存原始数据集
    Data = Merge([InputData,Result],['平均房间数目','房价'])
    Data.to_excel('./原始数据.xlsx')
    # 改变数据集与真实房价数组的形状
    InputData = InputData.reshape((len(InputData), 1))
    Result = np.array(Result).reshape((len(Result), 1))

    # 解决Matplotlib中的中文乱码问题，以便于后面实验结果可视化
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 把数据集分成训练数据集和测试数据集，测试集只占总数据集的10%
    train_data,test_data,train_result,test_result = \
        train_test_split(InputData,Result,test_size=0.1,random_state=10)

    # 利用散点图可视化测试数据集，并保存可视化结果
    col = ['真实房价']
    plt.scatter(test_data,test_result,alpha=0.5,c='b',s=10)
    plt.grid(True)
    plt.legend(labels = col,loc='best')
    plt.xlabel("房间数")
    plt.ylabel("真实房价")
    plt.savefig("./测试集可视化.jpg",bbox_inches='tight')
    #plt.show()
    plt.close()

    # 对测试数据集及其真实房价输出进行排序
    index = np.argsort(test_data.T[0])
    test_data_ = test_data[index]
    test_result_ = test_result[index]

    # 构建线性回归模型与局部加权线性回归模型
    # 定义带宽系数
    K = [0.001,0.01,0.1,0.5]
    K.extend(list(np.arange(1,201)))
    K = np.array(K)
    # 定义正则方程优化的线性回归模型
    linearregression = LinearRegression(train_data,train_result)
    # 定义局部加权线性回归模型
    lwlr = LocalWeightedLinearRegression(train_data,train_result)

    # 利用测试数据集进行预测
    # 线性回归利用正规方程求解模型参数
    linearregression.getNormalEquation()
    # 得到线性回归模型最佳参数之后，利用测试数据集进行预测，
    # 预测结果保存在predict_linearregression,并计算线性回归的预测均方误差loss_LR
    predict_LR = linearregression.predict(test_data_)
    loss_LR = ((predict_LR-test_result_)**2).T[0]
    #print(np.shape(predict_LR))
    #print(np.shape(loss_LR))
    # 由于局部加权回归算法对于每一组测试数据集，其权重都不一样。因此
    # 局部加权回归的最佳模型参数与测试数据相关，因此遍历测试数据集的同时
    # 求解最佳参数与预测同时进行。
    # 利用测试数据集，进行局部加权线性回归预测，回归预测结果保存
    # 在predict_LWLR,预测误差为loss_LWLR
    predict_LWLR = []
    loss_LWLR = []
    for k in K:
        predict = lwlr.predict_NormalEquation(test_data_,k)
        #print(np.shape(predict))
        predict_LWLR.append(predict)
        loss_LWLR.append(((predict-test_result_.T[0])**2))
    #print(np.shape(predict_LWLR))
    #print(np.shape(loss_LWLR))
 
    # 不同带宽系数的局部加权线性回归和线性回归模型预测结果的可视化
    plt.scatter(test_data,test_result,alpha=0.5,c='b',s=10)
    plt.grid(True)
    plt.plot(test_data_.T[0], predict_LR,'r')
    plt.legend(labels = ['线性回归'],loc='best')
    plt.xlabel("房间数")
    plt.ylabel("房价")
    plt.savefig("./测试集可视化"+"线性回归.jpg",bbox_inches='tight')
    plt.show()
    plt.close()
    # 遍历每组不同局部加权线性回归算法的预测误差
    for (predict_lwlr,k) in zip(predict_LWLR,K):
        plt.scatter(test_data, test_result, alpha=0.5, c='b', s=10)
        plt.plot(test_data_.T[0], predict_lwlr,'r')
        plt.grid(True)
        plt.legend(labels=["k="+str(k)], loc='best')
        plt.xlabel("房间数")
        plt.ylabel("房价")
        plt.savefig("./测试集可视化局部加权回归"+str(k)+".jpg", bbox_inches='tight')
        plt.close()

    # 可视化预测部分结果
    # 定义需要可视化的不同带宽系数的局部加权线性回归模型
    K_ = np.array([0.1,0.5,2,5,12,25,50,200])
    predict_LWLR_tmp = [predict_LWLR[3],predict_LWLR[4],predict_LWLR[6],
                        predict_LWLR[9],predict_LWLR[16],predict_LWLR[29],
                        predict_LWLR[54],predict_LWLR[203]]
    # 在第一个子图可视化线性回归预测结果
    fig = plt.figure()
    ax = fig.add_subplot(331)
    ax.scatter(test_data,test_result,alpha=0.5,c='b',s=10)
    ax.grid(True)
    ax.plot(test_data_, predict_LR,'r')
    ax.legend(labels = ['线性回归'],loc='best')
    plt.xlabel("房间数")
    plt.ylabel("房价")
    # 遍历局部加权线性回归算法的预测结果
    for (index,(predict_lwlr, k)) in enumerate(zip(predict_LWLR_tmp, K_)):
        # 在第index+1个子图可视化预测结果
        ax = fig.add_subplot(331 + index + 1)
        ax.scatter(test_data, test_result, alpha=0.5, c='b', s=10)
        ax.grid(True)
        ax.plot(test_data_.T[0], predict_lwlr, 'r')
        ax.legend(labels=['k='+str(k)], loc='best')
        plt.xlabel("房间数")
        plt.ylabel("房价")
        # 子图之间使用紧致布局
        plt.tight_layout()
    plt.savefig("./部分预测结果.jpg", bbox_inches='tight')
    plt.close()

    # 保存不同带宽系数的局部加权线性回归和线性回归模型预测结果
    # 定义列名称数组
    col = ['真实房价','LinearRegression']
    # 遍历带宽系数数组，补全列名称数组
    for k in K:
        col.append('K=' + str(k))
    data = [test_result_.T[0],predict_LR]
    # 遍历每种不同带宽系数的局部加权线性回归模型的预测结果
    for predict_lwlr in predict_LWLR:
        data.append(predict_lwlr)
    result = Merge(data,col)
    # 保存线性回归与局部加权线性回归预测结果为excel
    result.to_excel("./线性回归与局部加权线性回归预测对比.xlsx")
    # 保存线性回归与局部加权线性回归预测结果统计信息为excel
    result.describe().to_excel("./线性回归与局部加权线性回归预测对比统计信息.xlsx")

    # 计算两种模型的均方误差
    # 定义列名称数组
    col = ['LinearRegression']
    # 遍历带宽系数数组，补全列名称数组
    for k in K:
        col.append('K=' + str(k))
    # 定义线性回归与不用带宽系数的局部加权模型误差数组
    MSE = [loss_LR]
    # 遍历每种不同带宽系数的局部加权线性回归模型的预测结果
    for loss in loss_LWLR:
        MSE.append(loss)
    # 构造DataFrame数据
    result = Merge(MSE,col)
    # 保存线性回归与局部加权线性回归预测的均方误差为excel
    result.to_excel("./线性回归与局部加权线性回归预测的均方误差.xlsx")
    # 保存线性回归与局部加权线性回归预测的均方误差的统计信息为excel
    information = result.describe()
    information.to_excel("./线性回归与局部加权线性回归预测的均方误差对比统计信息.xlsx")

    # 可视化不同带宽参数的局部加权线性回归模型在测试集的均方误差和预测标准差
    K = list(np.arange(1, 201))
    col = ["LWLR-MSE", "LWLR-std"]
    LWLR_MSE = list(information.loc['mean'])[5:]
    LWLR_std = list(information.loc['std'])[5:]
    plt.plot(K, LWLR_MSE,'b')
    plt.plot(K, LWLR_std, 'c-.')
    plt.grid(True)
    plt.legend(labels=col, loc='best')
    plt.xlabel("带宽系数")
    plt.savefig("./局部加权线性回归的预测均方误差和标准差.jpg", bbox_inches='tight')
    #plt.show()
    plt.close()

if __name__ == '__main__':
    run_main()