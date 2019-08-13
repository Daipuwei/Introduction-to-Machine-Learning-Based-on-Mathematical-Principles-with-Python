#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/1610:19
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : PredictConcreteCompressiveStrength.py
# @Software: PyCharm

from LinearRegression.LinearRegression import LinearRegression
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

    # 把数据集分成训练数据集和测试数据集
    train_data,test_data,train_result,test_result = \
        train_test_split(InputData,Result,test_size=0.1,random_state=50)

    # 解决Matplotlib中的中文乱码问题，以便于后面实验结果可视化
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 利用散点图可视化测试数据集，并保存可视化结果
    col = ['真实房价']
    plt.scatter(test_data,test_result,alpha=0.5,c='b',s=10)
    plt.grid(True)
    plt.legend(labels = col,loc='best')
    plt.xlabel("房间数")
    plt.ylabel("真实房价")
    plt.savefig("./测试集可视化.jpg",bbox_inches='tight')
    plt.show()
    plt.close()

    # 开始构建线性回归模型
    col = np.shape(train_data)[1]+1
    # 初始化线性回归参数theta
    theta = np.random.random((col,1))
    # BGD优化的线性回归模型
    linearregression_BGD = LinearRegression(train_data, train_result,theta)
    # SGD优化的线性回归模型
    linearregression_SGD = LinearRegression(train_data, train_result,theta)
    # MBGD优化的线性回归模型
    linearregression_MBGD = LinearRegression(train_data, train_result,theta)
    # 正则方程优化的线性回归模型
    linearregression_NormalEquation = LinearRegression(train_data, train_result,theta)

    # 训练模型
    iter = 30000             # 迭代次数
    alpha = 0.001            # 学习率
    batch_size = 64          # 小样本规模
    # BGD的训练损失
    BGD_train_cost = linearregression_BGD.train_BGD(iter,alpha)
    # SGD的训练损失
    SGD_train_cost = linearregression_SGD.train_SGD(iter,alpha)
    # MBGD的训练损失
    MBGD_train_cost = linearregression_MBGD.train_MBGD(iter,batch_size,alpha)
    # 利用正规方程获取参数
    linearregression_NormalEquation.getNormalEquation()

    # 三种梯度下降算法迭代训练误差结果可视化，并保存可视化结果
    col = ['BGD','SGD','MBGD']
    iter = np.arange(iter)
    plt.plot(iter, BGD_train_cost, 'r-.')
    plt.plot(iter, SGD_train_cost, 'b-')
    plt.plot(iter, MBGD_train_cost, 'k--')
    plt.grid(True)
    plt.xlabel("迭代次数")
    plt.ylabel("平均训练损失")
    plt.legend(labels = col,loc = 'best')
    plt.savefig("./三种梯度算法的平均训练损失.jpg",bbox_inches='tight')
    plt.show()
    plt.close()

    # 三种梯度下降算法的训练损失
    # 整合三种梯度下降算法的训练损失到DataFrame
    train_cost = [BGD_train_cost,SGD_train_cost,MBGD_train_cost]
    train_cost = Merge(train_cost,col)
    # 保存三种梯度下降算法的训练损失及其统计信息
    train_cost.to_excel("./三种梯度下降算法的训练训练损失.xlsx")
    train_cost.describe().to_excel("./三种梯度下降算法的训练训练损失统计.xlsx")

    # 计算4种调优算法下的拟合曲线
    x = np.arange(int(np.min(test_data)), int(np.max(test_data)+1))
    x = x.reshape((len(x),1))
    # BGD算法的拟合曲线
    BGD = linearregression_BGD.test(x)
    # SGD算法的拟合曲线
    SGD = linearregression_SGD.test(x)
    # MBGD算法的拟合曲线
    MBGD = linearregression_MBGD.test(x)
    # 正则方程的拟合曲线
    NormalEquation = linearregression_NormalEquation.test(x)

    # 4种模型的拟合直线可视化，并保存可视化结果
    col = ['BGD', 'SGD', 'MBGD', '正则方程']
    plt.plot(x, BGD,'r-.')
    plt.plot(x, SGD, 'b-')
    plt.plot(x, MBGD, 'k--')
    plt.plot(x, NormalEquation, 'g:',)
    plt.scatter(test_data,test_result,alpha=0.5,c='b',s=10)
    plt.grid(True)
    plt.xlabel("房间数")
    plt.ylabel("预测值")
    plt.legend(labels = col,loc = 'best')
    plt.savefig("./预测值比较.jpg",bbox_inches='tight')
    plt.show()
    plt.close()

    # 利用测试集进行线性回归预测
    # BGD算法的预测结果
    BGD_predict = linearregression_BGD.test(test_data)
    # SGD算法的预测结果
    SGD_predict = linearregression_SGD.test(test_data)
    # MBGD算法的预测结果
    MBGD_predict = linearregression_MBGD.test(test_data)
    # 正则方程的预测结果
    NormalEquation_predict = linearregression_NormalEquation.test(test_data)

    # 保存预测数据
    # A.tolist()是将numpy.array转化为python的list类型的函数，是将A的所有元素
    # 当作一个整体作为list的一个元素，因此我们只需要A.tolist()的第一个元素
    data = [test_data.T.tolist()[0],test_result.T.tolist()[0],BGD_predict,
                SGD_predict,MBGD_predict,NormalEquation_predict]
    col = ["平均房间数目","真实房价",'BGD预测结果',
            'SGD预测结果','MBGD预测结果','正规方程预测结果']
    Data = Merge(data,col)
    Data.to_excel('./测试数据与预测结果.xlsx')

    # 计算4种算法的均方误差以及其统计信息
    # test_result之前的形状为(num,1)，首先计算其转置后
    # 获得其第一个元素即可
    test_result = test_result.T[0]
    # BGD算法的测试均方误差
    BGD_error = ((BGD_predict-test_result)**2)
    # SGD算法的测试均方误差
    SGD_error = ((SGD_predict-test_result)**2)
    # MBGD算法的测试均方误差
    MBGD_error = ((MBGD_predict-test_result)**2)
    # 正则方程的测试均方误差
    NormalEquation_error = ((NormalEquation_predict-test_result)**2)
    # 整合四种算法的测试均方误差到DataFrame
    error = [BGD_error,SGD_error,MBGD_error,NormalEquation_error]
    col = ['BGD', 'SGD', 'MBGD', '正则方程']
    error = Merge(error,col)
    # 保存四种测试均方误差及其统计信息
    error.to_excel("./四种算法的均方预测误差原始数据.xlsx")
    error.describe().to_excel("./四种算法的均方预测误差统计.xlsx")

if __name__ == '__main__':
    run_main()