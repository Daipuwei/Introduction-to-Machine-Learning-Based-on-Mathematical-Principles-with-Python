#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/3 9:57
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-506实验室
# @File    : KMeansCluster.py
# @Software: PyCharm

import numpy as np

"""
    这是K-Means聚类的Python类代码
"""

class KMeansCluster:
    def __init__(self,Data):
        """
        K-Means聚类算法的构造函数
        :param Data: 数据集
        :param k: 聚类簇数
        :param centroids: 聚类质心数组
        """
        self.Data = Data                    # 数据集
        row = np.shape(Data)[0]
        self.Label = np.array([0]*row)      # 数据集聚类标签
        self.centroids = []                 # 聚类质心数组

    def EcludDistance(self,dataA,dataB):
        """
        计算欧式距离
        :param dataA: 数据A
        :param dataB: 数据B
        """
        return np.sqrt(np.sum((dataA-dataB)**2))

    def Initial_Centroids(self,k,centroids = None):
        """
        这是初始化聚类质心的函数
        :param k: 聚类簇数
        :param centroids: 聚类质心数组
        :return:
        """
        # 初始化k
        self.K = k
        # centroids为None时，将聚类质心初始化为数据集中
        # 最小数据与最大数据之间的随机数
        if centroids is None:
            col = np.shape(self.Data)[1]
            self.centroids = np.zeros((k, col))  # 质心坐标
            # 开始初始化质心坐标
            for i in range(col):
                # 获取数据集中的最小值与最大值
                Min = np.min(self.Data[:, i])
                Max = np.max(self.Data[:, i])
                # 初始化聚类簇质心为最小值与最大值之间的随机数
                self.centroids[:, i] = Min + float(Max - Min) * np.random.rand(k)
        else:
            # centroids不为空则直接初始化为聚类质心
            self.centroids = centroids

    def cluster(self,k,centroids = None):
        """
        这是进行K-Means聚类算法的函数
        :param k: 聚类簇数
        :param centroids: 聚类质心数组
        """
        # 初始化聚类质心
        self.Initial_Centroids(k,centroids)
        # 开始执行K-Means聚类算法
        newdist = 0       # 当前迭代的距离之和
        olddist = 1    # 前一次迭代的距离之和
        # dist与olddist之间的差值小于1E-6结束循环
        while np.abs(newdist-olddist) > 1E-6:
            #print(self.centroids)
            # 将dist赋值给olddist
            olddist = newdist
            # 更新每组训练数据的标签
            for (i,data) in enumerate(self.Data):
                # 初始化每组数据与每个质心之间的距离数组
                _dist = []
                # 计算每组数据与质心之间的距离
                for centroid in self.centroids:
                    _dist.append(self.EcludDistance(data,centroid)**2)
                # 选择距离最小对应的质心进行作为这组数据的聚类标签
                self.Label[i] = np.argmin(_dist)
            # 更新聚类质心的坐标
            for i in range(self.K):
                # 获取聚类标签为i的子数据集
                cluster_data = self.Data[self.Label == i]
                size = len(cluster_data)
                # 防止出现聚类标签为i的子数据集为空的情况
                if size != 0:
                    self.centroids[i] = np.sum(cluster_data,0)/size
            # 初始化当前距离之和为0
            newdist = 0
            # 遍历整个数据集及其聚类标签集，计算每组数据与
            # 对应聚类簇质心之间的距离
            for (data,label) in zip(self.Data,self.Label):
                newdist += self.EcludDistance(data,self.centroids[label])**2
        return self.Label,self.centroids,newdist