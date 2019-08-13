#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/24 20:00
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25实验室506
# @File    : Other.py
# @Software: PyCharm

import pandas as pd
import numpy as np

def Merge(data,row,col):
    """
    这是生成DataFrame数据的函数
    :param data: 数据，格式为列表(list),不是numpy.array
    :param row: 行名称
    :param col: 列名称
    """
    data = np.array(data).T
    return pd.DataFrame(data=data,columns=col,index=row)

def Confusion_Matrix_Merge(confusion_matrix,name):
    return pd.DataFrame(data=confusion_matrix,index=name,columns=name)

def confusion_matrix(real_result,predict_result):
    """
    这是计算预测结果的混淆矩阵的函数
    :param real_result: 真实结果
    :param predict_result: 预测结果
    """
    labels = []
    for result in real_result:
        if result not in labels:
            labels.append(result)
    labels = np.sort(labels)
    #print(labels)
    # 计算混淆矩阵
    confusion_matrix = []
    for label1 in labels:
        # 真实结果中为label1的数据下标
        index = real_result == label1
        '''print("=====:\n")
        print(index)
        print("=====:\n")'''
        _confusion_matrix = []
        for label2 in labels:
            _predict_result = predict_result[index]
            '''print("....................")
            print(_predict_result)
            print(np.sum(_predict_result == label2))
            print("....................")'''
            _confusion_matrix.append(np.sum(_predict_result == label2))
        confusion_matrix.append(_confusion_matrix)
    confusion_matrix = np.array(confusion_matrix)
    return confusion_matrix

def Transform(Label):
    """
    这是将one-hot编码标签转化为数值标签的函数
    :param Label: one-hot标签
    """
    _Label = []
    for label in Label:
        _Label.append(np.argmax(label))
    return np.array(_Label)