#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/3 15:15
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-0506实验室
# @File    : IrisCluster.py
# @Software: PyCharm

from KMeansCluster.KMeansCluster import KMeansCluster
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

def run_main():
    """
       这是主函数
    """
    # Iris导入数据集
    Data,Label = load_iris(return_X_y=True)

    # 对数据进行最小最大标准化
    Data = MinMaxScaler().fit_transform(X = Data,y = None)

    # 解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 可视化Iris数据集
    col = np.shape(Data)[1]
    Col = ["花萼长度","花萼宽度","花瓣长度","花瓣宽度"]
    for i in range(0,col-1):
        for j in range(i+1,col):
            plt.scatter(Data[:,i],Data[:,j])
            plt.grid(True)
            plt.xlabel(Col[i])
            plt.ylabel(Col[j])
            plt.tight_layout()
            plt.savefig("./标准化数据集可视化Iris/"+str(i)+"-"+str(j)+".jpg", bbox_inches='tight')
            plt.close()

    # 构建K-Means聚类算法类
    K = np.arange(1,11)
    Dist = []
    Labels = []
    Centroids = []
    for k in K:
        kmeans = KMeansCluster(Data=Data)
        Label,centroids,dist = kmeans.cluster(k,None)
        Dist.append(dist)
        Labels.append(Label)
        Centroids.append(centroids)
        print("k=%d下的质心：" %(k))
        print(Centroids)
    plt.plot(K,Dist)
    plt.grid(True)
    plt.xlabel("k")
    plt.ylabel("聚类后的距离")
    plt.savefig("./性能对比Iris.jpg",bbox_inches='tight')
    plt.close()

    # 对k=3的聚类结果进行可视化
    # 遍历所有数据及其聚类标签
    colors = ['r','g',"b"]
    markers = ['o','*','x']
    for i in np.arange(0,col-1):
        for j in np.arange(i + 1, col):
            # 画每簇数据
            for (index, (c, m)) in enumerate(zip(colors, markers)):
                data = Data[Labels[2] == index]
                plt.scatter(data[:, i], data[:, j], c=c, marker=m, alpha=0.5)
            # 画聚类质心
            for centroid in Centroids[2]:
                plt.scatter(centroid[i], centroid[j], c="k", marker="+",s=100)
            # 画面属性设置
            plt.xlabel(Col[i])
            plt.ylabel(Col[j])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("./k=3聚类结果Iris/" + str(i) + "-" + str(j) + ".jpg", bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    run_main()