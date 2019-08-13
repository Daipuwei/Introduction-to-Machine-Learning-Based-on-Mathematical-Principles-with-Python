#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/10 21:56
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-506实验室
# @File    : ClusterWine.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from GMM.GMM import GMM
from KMeansCluster.KMeansCluster import KMeansCluster
from .Load_Wine import Load_Wine
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.preprocessing import StandardScaler

def run_main():
    """
        这是主函数
    """
    # 导入葡萄酒数据集
    path = "./Wine.txt"
    Data,Label = Load_Wine(path)

    # 对数据进行分标准归一化
    Data = StandardScaler().fit_transform(Data)

    # 解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 可视化葡萄酒数据集
    col = np.shape(Data)[1]
    Col = ["酒精","苹果酸","灰","灰烬的碱度","镁","总酚",
           "黄酮类化合物","类黄酮酚","原花青素","颜色强度",
           "色调","OD280/OD315","脯氨酸"]
    for i in range(0,col-1):
        for j in range(i+1,col):
            plt.scatter(Data[:,i],Data[:,j])
            plt.grid(True)
            plt.xlabel(Col[i])
            plt.ylabel(Col[j])
            plt.tight_layout()
            plt.savefig("./标准化数据集可视化Wine/"+str(i)+"-"+str(j)+".jpg", bbox_inches='tight')
            plt.close()

    # 比较GMM与K-Means的聚类结果
    # 初始化聚类种类数
    K = list(np.arange(2,11))
    # ARI系数数组
    kmeans_ari = []
    gmm_ari = []
    # 调整互信息数组
    kmeans_ami = []
    gmm_ami = []
    # silhouette系数数组
    kmeans_silhouette = []
    gmm_silhouette = []
    # calinski_harabaz系数数组
    kmeans_calinski_harabaz = []
    gmm_calinski_harabaz = []
    # 聚类结果数组
    kmeans_Labels = []
    gmm_Labels = []
    for k in K:
        # 构建GMM模型
        gmm = GMM(Data,k)
        kmeans = KMeansCluster(Data)
        # 利用GMM进行聚类
        kmeans_cluster_labels,_,__ = kmeans.cluster(k,None)
        gmm_cluster_labels,_ = gmm.GMM_EM()
        # 保存聚类结果
        kmeans_Labels.append(kmeans_cluster_labels)
        gmm_Labels.append(gmm_cluster_labels)
        # 计算ARI系数
        kmeans_ari.append(adjusted_rand_score(Label, kmeans_cluster_labels))
        gmm_ari.append(adjusted_rand_score(Label, gmm_cluster_labels))
        # 计算调整互信息
        kmeans_ami.append(adjusted_mutual_info_score(Label, kmeans_cluster_labels))
        gmm_ami.append(adjusted_mutual_info_score(Label,gmm_cluster_labels))
        # 计算silhouette
        kmeans_silhouette.append(silhouette_score(Data, kmeans_cluster_labels, metric='euclidean'))
        gmm_silhouette.append(silhouette_score(Data,gmm_cluster_labels,metric='euclidean'))
        # 计算calinski_harabaz
        kmeans_calinski_harabaz.append(calinski_harabaz_score(Data,kmeans_cluster_labels))
        gmm_calinski_harabaz.append(calinski_harabaz_score(Data, gmm_cluster_labels))

    # 对ARI、AMI与V-Measure系数的进行可视化（2张子图）
    fig = plt.figure()      # 生成画布
    legend = ["K-Means", "GMM"]
    # ARI对比结果可视化
    ax = fig.add_subplot(211)   #加入第一个子图
    ax.plot(K, kmeans_ari, 'r-')
    ax.plot(K, gmm_ari, 'b--')
    ax.grid(True)
    plt.xlabel("K")
    plt.ylabel("ARI")
    plt.legend(labels=legend, loc="best")
    plt.title("ARI系数对比")
    plt.tight_layout()
    # AMI对比结果可视化
    ax = fig.add_subplot(212)  # 加入第二个子图
    ax.plot(K, kmeans_ami, 'r-')
    ax.plot(K, gmm_ami, 'b--')
    ax.grid(True)
    plt.xlabel("K")
    plt.ylabel("AMI")
    plt.legend(labels=legend, loc="best")
    plt.title("AMI系数对比")
    plt.tight_layout()
    plt.savefig("./ARI和AMI系数.jpg", bbox_inches='tight')
    plt.close()

    # calinski_harabaz和silhouette系数对比可视化（2张子图）
    fig = plt.figure()  # 生成画布
    # calinski_harabaz对比结果可视化
    ax = fig.add_subplot(211)  # 加入第一个子图
    ax.plot(K, kmeans_calinski_harabaz, 'r-')
    ax.plot(K, gmm_calinski_harabaz, 'b--')
    ax.grid(True)
    plt.xlabel("K")
    plt.ylabel("calinski_harabaz")
    plt.legend(labels=legend, loc="best")
    plt.title("calinski_harabaz系数对比")
    plt.tight_layout()
    # silhouette对比结果可视化
    ax = fig.add_subplot(212)  # 加入第二个子图
    ax.plot(K, kmeans_silhouette, 'r-')
    ax.plot(K, gmm_silhouette, 'b--')
    ax.grid(True)
    plt.xlabel("K")
    plt.ylabel("silhouette")
    plt.legend(labels=legend, loc="best")
    plt.tight_layout()
    plt.title("silhouette系数对比")
    plt.savefig("./calinski_harabaz和silhouette系数.jpg", bbox_inches='tight')
    plt.close()

    # ARI对比结果可视化
    legend = ["K-Means","GMM"]
    plt.plot(K, kmeans_ari, 'r-')
    plt.plot(K, gmm_ari, 'b--')
    plt.xlabel("K")
    plt.ylabel("ARI")
    plt.grid(True)
    plt.legend(labels = legend,loc = "best")
    plt.savefig("./ARI系数.jpg", bbox_inches='tight')
    plt.close()

    # 调整互信息对比结果可视化
    plt.plot(K, kmeans_ami, 'r-')
    plt.plot(K, gmm_ami, 'b--')
    plt.xlabel("K")
    plt.ylabel("AMI")
    plt.grid(True)
    plt.legend(labels = legend,loc = "best")
    plt.savefig("./AMI系数.jpg", bbox_inches='tight')
    plt.close()

    # calinski_harabaz对比结果可视化
    plt.plot(K, kmeans_calinski_harabaz, 'r-')
    plt.plot(K, gmm_calinski_harabaz, 'b--')
    plt.xlabel("K")
    plt.ylabel("calinski_harabaz")
    plt.grid(True)
    plt.legend(labels = legend,loc = "best")
    plt.savefig("./calinski_harabaz系数.jpg", bbox_inches='tight')
    plt.close()

    # silhouette对比结果可视化
    plt.plot(K, kmeans_silhouette, 'r-')
    plt.plot(K, gmm_silhouette, 'b--')
    plt.xlabel("K")
    plt.ylabel("silhouette")
    plt.grid(True)
    plt.legend(labels = legend,loc = "best")
    plt.savefig("./silhouette系数.jpg", bbox_inches='tight')
    plt.close()

    # 可视化最优聚类结果
    # 对k=3的聚类结果进行可视化
    # 遍历所有数据及其聚类标签
    colors = ['r','g',"b"]
    markers = ['o','*','x']
    for i in np.arange(0,col-1):
        for j in np.arange(i + 1, col):
            # 画每簇数据
            for (index, (c, m)) in enumerate(zip(colors, markers)):
                data = Data[gmm_Labels[2] == index]
                plt.scatter(data[:, i], data[:, j], c=c, marker=m, alpha=0.5)
            # 画面属性设置
            plt.xlabel(Col[i])
            plt.ylabel(Col[j])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("./k=3聚类结果Wine/" + str(i) + "-" + str(j) + ".jpg", bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    run_main()