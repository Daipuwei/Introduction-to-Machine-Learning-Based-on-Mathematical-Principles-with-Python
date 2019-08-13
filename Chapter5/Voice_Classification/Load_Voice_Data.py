#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 8:55
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 
# @File    : Load_Voice_Data.py
# @Software: PyCharm

import numpy as np

def Load_Voice_Data(path):
    """
    这是导入数据的函数
    :param path: 数据文件的路径
    :return: 数据集
    """
    data = []
    label = []
    with open(path) as f:
        # 遍历文件中每一行
        for line in f.readlines():
            # 根据"Tab"进行分割整个字符串
            str = line.strip().split("\t")
            tmp = []
            # 获取每组数据，从第二个开始
            for i in range(1,len(str)):
                tmp.append(float(str[i]))
            data.append(tmp)
            # 第一个字符属于分类标签，得到对应的one-hot标签
            if 1 == int(str[0]):
                label.append([1,0,0,0])
            elif 2 == int(str[0]):
                label.append([0,1,0,0])
            elif 3 == int(str[0]):
                label.append([0,0,1,0])
            else:
                label.append([0,0,0,1])
    data = np.array(data)
    label = np.array(label)
    return data,label