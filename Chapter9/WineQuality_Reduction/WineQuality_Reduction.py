#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 15:03
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# blog     : https://blog.csdn.net/qq_30091945
# @Site    : 中国民航大学北教25-506实验室
# @File    : WineQuality_Reduction.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PCA.PCA import PCA
from .Load_WineQuality import Load_WineQuality

def Merge(data,row,col):
    """
    这是生成DataFrame数据的函数
    :param data: 数据，格式为列表(list),不是numpy.array
    :param row: 行名称
    :param col: 列名称
    """
    data = np.array(data).T
    return pd.DataFrame(data=data,columns=col,index=row)

def run_main():
    """
       这是主函数
    """
    # 导入红葡萄酒数据集
    red_winequality_path = "./winequality-red.csv"
    Red_WineQuality_Data, _ = Load_WineQuality(red_winequality_path)
    print(Red_WineQuality_Data)
    print(np.shape(Red_WineQuality_Data))

    # 导入白葡萄酒数据集
    white_winequality_path = "./winequality-white.csv"
    White_WineQuality_Data,__ = Load_WineQuality(white_winequality_path)
    print(White_WineQuality_Data)
    print(np.shape(White_WineQuality_Data))

    # 解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 可视化原始数据
    feature_name = ["fixed acidity","volatile acidity","citric acid","residual sugar",
                    "chlorides","free sulfur dioxide","total sulfur dioxide",
                    "density","pH","sulphates","alcohol"]
    # 红葡萄酒数据集可视化
    for i in range(len(feature_name)-1):
        for j in range(i+1,len(feature_name)):
            plt.scatter(Red_WineQuality_Data[:, i], Red_WineQuality_Data[:, j],s=5)
            plt.xlabel(feature_name[i])
            plt.ylabel(feature_name[j])
            plt.grid(True)
            plt.savefig("./红葡萄酒数据集可视化/" + str(i) + "-" + str(j) + ".jpg", bbox_inches='tight')
            plt.close()

    # 白葡萄酒数据集可视化
    for i in range(len(feature_name)-1):
        for j in range(i+1,len(feature_name)):
            plt.scatter(White_WineQuality_Data[:, i], White_WineQuality_Data[:, j],s=5)
            plt.xlabel(feature_name[i])
            plt.ylabel(feature_name[j])
            plt.grid(True)
            plt.savefig("./白葡萄酒数据集可视化/"+str(i)+"-"+str(j)+".jpg", bbox_inches='tight')
            plt.close()

    # 构建PCA类
    red_pca = PCA(Red_WineQuality_Data,feature_name)
    white_pca = PCA(White_WineQuality_Data,feature_name)
    # 对数据集进行PCA降维,获取方差百分比以及累积方差百分比
    red_var_percentage_pca,red_var_accumulation_percentage_pca,\
                           red_feature_name = red_pca.PCA_Reduction()
    white_var_percentage_pca, white_var_accumulation_percentage_pca,\
                           white_feature_name = white_pca.PCA_Reduction()

    # 对PCA降维的红葡萄酒数据方差百分比进行可视化
    plt.plot(np.arange(11),red_var_percentage_pca,"bx-")
    plt.xlabel("K")
    plt.ylabel("方差所占比例")
    plt.xticks(np.arange(0, 11), np.arange(1, 12))
    plt.grid(True)
    plt.savefig("./红葡萄酒数据属性方差百分比PCA.jpg", bbox_inches='tight')
    plt.close()

    # 对PCA降维的红葡萄酒数据方差累积百分比进行可视化
    plt.plot(np.arange(11),red_var_accumulation_percentage_pca,"bx-")
    plt.xlabel("K")
    plt.ylabel("方差累积所占比例")
    plt.xticks(np.arange(0, 11), np.arange(1, 12))
    plt.grid(True)
    plt.savefig("./红葡萄酒数据属性方差累积百分比PCA.jpg", bbox_inches='tight')
    plt.close()

    #保存PCA降维的红葡萄酒数据属性方差百分比和累积百分比
    data = [red_var_percentage_pca,red_var_accumulation_percentage_pca]
    col = ["方差所占比例","方差累积所占比例"]
    ans = Merge(data,red_feature_name,col)
    ans.to_excel("./红葡萄酒数据属性方差累积百分比PCA.xlsx")

    # 对PCA降维的白葡萄酒数据方差百分比进行可视化
    plt.plot(np.arange(11),white_var_percentage_pca,"bx-")
    plt.xlabel("K")
    plt.ylabel("方差所占比例")
    plt.xticks(np.arange(0, 11), np.arange(1, 12))
    plt.grid(True)
    plt.savefig("./白葡萄酒数据属性方差百分比PCA.jpg", bbox_inches='tight')
    plt.close()

    # 对PCA降维的白葡萄酒数据方差累积百分比进行可视化
    plt.plot(np.arange(11),white_var_accumulation_percentage_pca,"bx-")
    plt.xlabel("K")
    plt.ylabel("方差累积所占比例")
    plt.xticks(np.arange(0,11),np.arange(1,12))
    plt.grid(True)
    plt.savefig("./白葡萄酒数据属性方差累积百分比PCA.jpg", bbox_inches='tight')
    plt.close()

    #保存PCA降维的白葡萄酒数据属性方差百分比和累积百分比
    data = [white_var_percentage_pca,white_var_accumulation_percentage_pca]
    col = ["方差所占比例","方差累积所占比例"]
    ans = Merge(data,white_feature_name,col)
    ans.to_excel("./白葡萄酒数据属性方差累积百分比PCA.xlsx")

    # 对PCA降维的红葡萄酒降维数据集进行可视化
    size = 5
    Red_WineQuality_Data_Reduction_PCA = red_pca.get_ReductionData(size)
    for i in range(size-1):
        for j in range(i+1,size):
            plt.scatter(Red_WineQuality_Data_Reduction_PCA[:, i],
                        Red_WineQuality_Data_Reduction_PCA[:, j],s=5)
            plt.xlabel("主成分"+str(i+1))
            plt.ylabel("主成分"+str(j+1))
            plt.grid(True)
            plt.savefig("./红葡萄酒数据集降维可视化PCA/"+str(i)+"-"+str(j)+".jpg"
                         ,bbox_inches='tight')
            plt.close()

    # 对PCA降维的白葡萄酒数据集进行可视化
    size = 6
    White_WineQuality_Data_Reduction_PCA = white_pca.get_ReductionData(size)
    for i in range(size-1):
        for j in range(i+1,size):
            plt.scatter(White_WineQuality_Data_Reduction_PCA[:, i],
                        White_WineQuality_Data_Reduction_PCA[:, j],s=5)
            plt.xlabel("主成分"+str(i+1))
            plt.ylabel("主成分"+str(j+1))
            plt.grid(True)
            plt.savefig("./白葡萄酒数据集降维可视化PCA/"+str(i)+"-"+str(j)+".jpg"
                        ,bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    run_main()