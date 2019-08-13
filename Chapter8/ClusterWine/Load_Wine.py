#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/11 16:55
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-506实验室
# @File    : Load_Wine.py
# @Software: PyCharm

import numpy as np

def Load_Wine(path):
    """
    这是导入葡萄酒数据集的函数
    :param path: 文件路径
    """
    Data = []
    Label = []
    with open(path) as f:
        # 遍历每行数据
        for line in f.readlines():
            # 分割每行字符串
            strs = line.strip().split(',')
            # 第一个数据为标签
            Label.append(int(strs[0]))
            # 遍历剩下的数据
            tmp = []
            for str in strs[1:]:
                tmp.append(float(str))
            Data.append(tmp)
        Data = np.array(Data)
        Label = np.array(Label)
    return Data,Label