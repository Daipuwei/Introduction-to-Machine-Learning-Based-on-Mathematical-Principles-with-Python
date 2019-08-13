#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/10 21:53
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-506实验室
# @File    : GMM.py
# @Software: PyCharm

"""
    这是高斯混合模型的Python类代码，这份代码使用EM算法进行
    求解模型参数,对于高斯分布的均值向量利用K-Means聚类算法
    的最终的聚类质心代替，协方差矩阵利用数据集的协方差矩阵代替
"""

import numpy as np
from KMeansCluster.KMeansCluster import KMeansCluster

class GMM(object):
    def __init__(self,Data,K,weights = None,means = None,covars = None):
        """
        这是GMM（高斯混合模型）类的构造函数
        :param Data: 数据集
        :param K: 高斯分布的个数
        :param weigths: 每个高斯分布的初始概率（权重）数组，默认为None，即可以没有
        :param means: 高斯分布的均值向量数组，默认为None，即可以没有
        :param covars: 高斯分布的协方差矩阵数组，默认为None，即可以没有
        """
        # 初始化数据集、高斯分布个数和数据集的形状
        self.Data = Data
        self.K = K
        self.size,self.dim = np.shape(self.Data)
        # 初始化数据集中每组数据属于各个高斯分布的概率数组
        # possibility[i][j]代表第i个样本属于第j个高斯分布的概率
        self.possibility = np.array([list(np.zeros(self.K)) for i in range(self.size)])
        # 初始化聚类标签数组
        self.clusterlabels = []
        # 随机隐含变量的多项式分布的参数数组不为None则进行初始化
        if weights is not None:
            self.weights = weights
        else:  # 随机隐含变量的多项式分布的参数数组为None时
            self.weights  = np.array([1.0 / self.K] * self.K)
        # 高斯分布的均值向量数组不为None则进行初始化
        if means is not None:
            self.means = means
        else:   # 高斯分布的均值向量数组为None时
            # 获取高斯分布的均值向量数组
            self.means = self.get_means(self.Data,self.K)
        # 高斯分布的协方差矩阵数组不为None则进行初始化
        if covars is not None:
            self.covars = covars
        else:   # 高斯分布的协方差矩阵数组为None时
            # 利用数据集的协方差矩阵代替高斯分布的协方差矩阵
            self.covars = self.get_cov(self.Data,self.K)

    def get_means(self,data,K):
        """
        K-Means聚类算法的聚类质心作为高斯分布的均值向量
        :param data: 数据集
        :param K: 高斯分布个数
        :param criter: 标准系数
        """
        # 定义K-Means聚类算法
        kmeans = KMeansCluster(data)
        # 获取K-Means的聚类质心
        _,centroids,__ = kmeans.cluster(K,None)
        return centroids

    def get_cov(self,data,K):
        """
        这是生成矩阵的函数
        :param data: 数据集
        :param k： 高斯混合模型个数
        """
        covs = []
        for i in range(K):
            # 利用数据集的协方差矩阵作为高斯分布的协方差矩阵
            covs.append(np.cov(data,rowvar=False))
        return covs

    def Gaussian(self,x,mean,cov):
        """
        这是自定义高斯分布概率密度函数
        :param x: 输入数据
        :param mean: 均值数组
        :param cov: 协方差矩阵
        :return: x的概率
        """
        # 获取协方差矩阵规模，即数据维数
        dim = np.shape(cov)[0]
        # cov的行列式为零时,加上一个与协防矩阵同规模的单位阵乘较小的常数
        if np.linalg.det(cov) == 0:
            covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
            covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        else: # cov的行列式不为零时
            covdet = np.linalg.det(cov)
            covinv = np.linalg.inv(cov)
        # 计算数据与均值向量之间的的差值
        xdiff = x - mean
        xdiff = np.reshape(xdiff,(1,len(xdiff)))
        # 计算高斯分布概率密度值
        prob = 1.0/(np.power(2*np.pi,1.0*dim/2)*np.sqrt(np.abs(covdet)))* \
               np.exp(-0.5*np.dot(np.dot(xdiff,covinv),xdiff.T))[0][0]
        return prob

    def GMM_EM(self):
        """
        这是利用EM算法进行优化GMM参数的函数
        :return: 返回各组数据的属于每个分类的概率
        """
        loglikelyhood = 0       # 当前迭代的极大似然函数值
        oldloglikelyhood = 1    # 上一次迭代的极大似然函数值
        while np.abs(loglikelyhood-oldloglikelyhood) > 1E-4:
            oldloglikelyhood = loglikelyhood
            # 下面是EM算法的E-step
            # 遍历整个数据集，计算每组数据属于每个高斯分布的后验概率
            self.possibility = []
            for data in self.Data:
                # respons是E-Step中每组数据与对应的随机隐含变量之间的联合概率数组
                respons = np.array([self.weights[k] * self.Gaussian(data, self.means[k], self.covars[k])
                                                for k in range(self.K)])
                # 计算联合概率之和
                sum_respons = np.sum(respons)
                # 利用全概率公式计算每组数据对应于各个高斯分布的后验概率
                respons = respons / sum_respons
                self.possibility.append(list(respons))
            self.possibility = np.array(self.possibility)
            # 下面是EM算法的M-step，根据E-step设置的后验概率更新G模型参数
            # 遍历每个高斯分布
            for k in range(self.K):
                #计算数据集中每组数据属于第k个高斯分布的概率和
                sum_gaussionprob_k = np.sum([self.possibility[i][k] for i in range(self.size)])
                # 更新第k个高斯分布的均值向量
                self.means[k] = (1.0 / sum_gaussionprob_k) * np.sum([self.possibility[i][k] * self.Data[i]
                                                                     for i in range(self.size)], axis=0)
                # 计算数据集与均值向量之间的差值
                xdiffs = self.Data - self.means[k]
                # 更新第k个高斯分布的协方差矩阵
                self.covars[k] = (1.0/sum_gaussionprob_k)*\
                                 np.sum([self.possibility[i][k]*xdiffs[i].reshape(self.dim,1)*xdiffs[i]
                                            for i in range(self.size)],axis=0)
                # 更新随机隐含变量的多项式分布的第k个参数
                self.weights[k] = 1.0 * sum_gaussionprob_k / self.size
            # 更新整个数据集的极大似然函数值
            loglikelyhood = []
            # 遍历整个数据集,计算每组数据的对应的极大似然函数值
            for data in self.Data:
                # 遍历每个高斯分布，计算每组数据在每个高斯分布下的极大似然估计
                data_Likelyhood = [self.Likelyhood(data,k) for k in range(self.K)]
                loglikelyhood.extend(data_Likelyhood)
            # 计算最终的数据集的极大似然函数值
            loglikelyhood = np.log(self.Mul(np.array(loglikelyhood)))
        # 对每组数据集分配到各个高斯分布的概率进行归一化
        for i in range(self.size):
            self.possibility[i] = self.possibility[i]/np.sum(self.possibility[i])
        # 生成每组数据的聚类标签
        self.clusterlabels = np.array([np.argmax(_possibility) for _possibility in self.possibility])
        return self.clusterlabels,loglikelyhood

    def Mul(self,data):
        """
        这是进行数据连乘的函数
        :param data: 数组
        """
        ans = 1.0
        for _data in data:
            ans = ans * _data
        return ans

    def Likelyhood(self,data,k):
        """
        这是计算每组数据在第k个高斯分布下的的极大似然函数值
        :param data: 数据
        :param k: 第k个高斯分布
        """
        # 计算第k个高斯分布下的概率值
        gaussian = self.Gaussian(data, self.means[k], self.covars[k])
        # 数据在第k个高斯分布下的的极大似然函数值为第k个
        # 高斯分布下的概率值与多项式分布的第k的参数的乘积
        likelyhood = self.weights[k] * gaussian
        return likelyhood