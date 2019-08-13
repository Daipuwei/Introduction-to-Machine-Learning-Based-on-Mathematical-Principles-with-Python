#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/9 18:51
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : PredictAbalone.py
# @Software: PyCharm

from RidgeRegression.RidgeRegression import RidgeRegression
from LinearRegression.LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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
            #一组数据追加到数据集中
            data.append(tmp)
        data = np.array(data)
        result = np.array(result)
    return data,result

def Error(real_result,predict_result):
    error = (real_result-predict_result).T
    return error*error

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
    Data, Result = Load_Abalone(path)
    Result = np.reshape(Result,(len(Result),1))

    # 把数据集分成训练集合测试集
    Train_Data, Test_Data, Train_Result, Test_Result = train_test_split \
        (Data, Result, test_size=0.2, random_state=50)

    #初始化模型参数
    theta = np.random.randn(np.shape(Train_Data)[1]+1,1)

    # 构造线性回归
    linearregression = LinearRegression(Train_Data,Train_Result,theta)

    #进行迭代，获取线性回归的最佳参数
    lr_train_error = linearregression.train_MBGD(10000,64,0.001)
    lr_predict = linearregression.predict(Test_Data)
    lr_test_error = Error(Test_Result.T[0], lr_predict)
    print("线性回归的模型参数：\n", linearregression.Theta.T)
    print("预测误差平均值为：", np.average(lr_test_error))
    print("训练误差平均值为：",np.average(lr_train_error))

    #初始化Ridge回归的正则化系数
    lambds = [0.00001,0.00003,0.0001,0.0003,0.001,0.01]

    #构造Ridge回归，获取不同正则化系数下Ridge回归最佳模型参数并进行预测
    ridge_predicts = []         #初始化Ridge回归预测结果数组
    ridge_test_errors = []      #初始化Ridge回归测试误差数组
    ridge_train_errors = []     #初始化Ridge回归迭代训练损失数组
    #遍历所有正则化参数
    for lambd in lambds:
        ridgeregression = RidgeRegression(Train_Data, Train_Result,theta)
        ridge_train_error = ridgeregression.train_MBGD(10000,64,0.001,lambd)
        ridge_train_errors.append(ridge_train_error)
        ridge_result = ridgeregression.test(Test_Data)
        ridge_test_error = Error(Test_Result.T[0],ridge_result)
        ridge_predicts.append(ridge_result)
        ridge_test_errors.append(ridge_test_error)
        print("正则化系数为%f下的Ridge回归的模型参数:" %(lambd))
        print(ridgeregression.Theta.T)
        print("预测误差平均值为：",np.average(ridge_test_error))
        print("训练误差平均值为：", np.average(ridge_train_error))

    #迭代训练误差可视化,可视化一小部分
    col = ["线性回归"]
    train_error = [lr_train_error]
    for (lambd,ridge_train_error) in zip(lambds,ridge_train_errors):
        col.append("lambda="+str(lambd))
        train_error.append(ridge_train_error)
    train_error = Merge(train_error,col)
    train_error.to_excel("./线性回归和Ridge回归的迭代训练误差.xlsx")
    train_error.describe().to_excel("./线性回归和Ridge回归的迭代训练误差的统计信息.xlsx")

    #计算预测值与真实之间的标准差
    test_error = []
    test_error.append(lr_test_error)
    for ridge_test_error in ridge_test_errors:
        test_error.append(ridge_test_error)
    test_error = Merge(test_error,col)
    test_error.to_excel("./模型的预测误差.xlsx")
    test_error.describe().to_excel("./模型的预测误差统计信息.xlsx")

if __name__ == '__main__':
    run_main()