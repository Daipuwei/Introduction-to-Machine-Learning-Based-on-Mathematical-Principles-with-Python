#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/9 14:56
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-506实验室
# @File    : Activation_Function.py
# @Software: PyCharm

'''
    这是几种激活函数及其导数的相关定义
'''

import numpy as np

def Sigmoid(x):
    """
    这是S型激活函数计算公式，在函数中我们必须
    防止数据上溢出与下溢出两种情况
    :param x: 需要进行计算的数据
    :return: S型激活函数的函数值
    """
    function = 1.0 / (1.0 + np.exp(-x))
    return function

def Sigmoid_Derivative(x):
    """
    这是S型激活函数的导数计算公式
    :param x: 需要进行计算的数据
    :return: S型激活函数的导数的函数值
    """
    f = Sigmoid(x)
    derivative = f*(1.0-f)
    return derivative

def ReLU(x):
    """
    这是ReLU激活函数的计算公式
    :param x:输入的数据
    :return:返回tanh激活函数的函数值
    """
    result = []
    col = np.shape(x)[1]
    for _x in x:
        r = np.zeros(col)
        for j in range(col):
            if _x[j] > 0.0:
                r[j] = _x[j]
        result.append(r)
    result = np.array(result)
    return  result

def ReLU_Derivative(x):
    """
    这是S型激活函数的导数计算公式
    :param x: 需要进行计算的数据
    :return: S型激活函数的导数的函数值
    """
    derivative = []
    col = np.shape(x)[0]
    for _x in x:
        d = np.zeros(col)
        for j in range(col):
            if _x[j] > 0:
                d[j] = 1.0
        derivative.append(d)
    derivative = np.array(derivative)
    return derivative

def Tanh(x):
    """
    这是Tanh激活函数计算公式
    :param x: 需要进行计算的数据
    :return: Tanh激活函数的函数值
    """
    function = 2.0 / (1.0 + np.exp(-2*x))-1
    return function

def Tanh_Derivative(x):
    """
    这是Tanh激活函数的导数计算公式
    :param x: 需要进行计算的数据
    :return: Tanh激活函数的导数的函数值
    """
    f = Tanh(x)
    derivative = 1.0-f*f
    return derivative

def Softmax(x):
    """
    这是Softmax激活函数的计算公式。Softmax函数有一个有效的性质，
    输入向量减去一个数之后不会影响输入结果。考虑到自变量大于1000时，
    指数函数会出现指数爆炸情况，即函数值会变得非常庞大，
    这可能会导致该数值在计算机中数值溢出。因此在实际Softmax
    函数的编写中，我们通常会利用广播的形式对每个分量减去最大分量，
    之后再计算相应的Softmax函数值，用来避免计算机内中数值溢出。
    :param x: 输入向量
    """
    # 获取向量中最大分量
    x_max = np.max(x)
    # 利用广播的形式对每个分量减去最大分量
    input = x - x_max
    # 计算向量指数运算后的和
    sum = np.sum(np.exp(input))
    # 计算Softmax函数值
    ans = np.exp(input)/sum
    return ans