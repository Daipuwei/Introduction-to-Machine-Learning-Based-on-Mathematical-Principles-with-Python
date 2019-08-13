#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 12:39
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25—506实验室
# @File    : PCA.py
# @Software: PyCharm

import numpy as np

class PCA(object):
    def __init__(self,Data,feature_name = None):
        """
        这是主成分分析（PCA）的构造函数
        :param Data: 输入数据
        :param feature_name: 特征名，默认为None，即可以没有
        """
        # 计算数据集的均值向量
        mean = np.average(Data,0)
        # 计算数据集的标准差
        std = np.std(Data, axis=0)
        # 对数据进行标准化
        self.Data = (Data-mean)/std
        # 初始化特征值与特征向量
        self.Eigenvalues = []
        self.Eigenvectors = []
        # 初始化方差累积百分比
        self.var_accumulation_percentage = []
        # 初始化方差百分比
        self.var_percentage = []
        # 初始化降维数据集
        self.Data_Reduction = []
        # 初始化数据集的协方差矩阵
        self.cov = []
        # 初始化特征名称
        if feature_name is None:
            self.feature_name = np.arange(1,np.shape(Data)[1]+1)
        else:
            self.feature_name = np.array(feature_name)

    def PCA_Reduction(self):
        """
        这是主成分分析（PCA）的降维过程的函数
        """
        # 计算数据集属性的协方差矩阵
        self.cov = np.cov(self.Data, rowvar=False)
        # 计算协方差矩阵的特征值与特征向量,由于np.linalg.eig
        self.Eigenvalues,self.Eigenvectors = np.linalg.eig(self.cov)
        # 遍历每个特征向量，将特征向量单位化
        for i,eigenvector in enumerate(self.Eigenvectors):
            sum = np.sum(eigenvector)
            self.Eigenvectors[i] = eigenvector/sum
        # 对特征值与特征向量进行从大到小进行排序
        order = np.argsort(-self.Eigenvalues)       # 获取从大到小序列
        self.Eigenvalues = self.Eigenvalues[order]
        self.Eigenvectors = self.Eigenvectors[order]
        self.feature_name = self.feature_name[order]
        # 计算数据属性的方差累积百分比
        sum = np.sum(self.Eigenvalues)
        self.var_accumulation_percentage = np.cumsum(self.Eigenvalues/sum)
        # 计算数据属性的方差百分比
        self.var_percentage = self.Eigenvalues/sum
        return self.var_percentage,self.var_accumulation_percentage,list(self.feature_name)

    def get_ReductionData(self,k):
        """
        这是获取降维后的数据的函数
        :param k: 候选主成分个数
        """
        # 对数据进行降维，遍历每组数据
        for data in self.Data:
            data_redution = []
            # 遍历前k个特征值与单位特征向量，计
            # 算每组数据在单位特征向量上的投影
            for i in np.arange(k):
                # 计算数据在特征向量上的投影
                data_projection = self.Eigenvectors[i].dot(data)
                data_redution.append(data_projection)
            # 将降维后的数据追加到降维数据集中
            self.Data_Reduction.append(data_redution)
        self.Data_Reduction = np.array(self.Data_Reduction)
        return self.Data_Reduction