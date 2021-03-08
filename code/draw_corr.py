import xlrd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from pyheatmap.heatmap import HeatMap
from sklearn import linear_model
import matplotlib.pyplot as plt  # 导入绘图包matplotlib（需要安装，方法见第一篇笔记）
import time
import csv
import codecs
import pandas as pd
import numpy as np
# print(pd.read_csv('file.tsv', delimiter='t'))
import datetime
from textblob import TextBlob
from scipy.stats import norm
import math
import itertools
from itertools import product
# from compiler.ast import flatten
from scipy import stats
import collections
import networkx as nx
from sklearn.linear_model import LinearRegression

from sklearn.metrics.pairwise import cosine_similarity, paired_distances

from scipy import spatial


def distance_euclidean_scipy(vec1, vec2, distance="euclidean"):
    return spatial.distance.cdist(vec1, vec2, distance)


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom

    return cos


def excel_to_matrix(path, fid):
    table = xlrd.open_workbook(path).sheets()[fid]  # 获取第一个sheet表
    row = table.nrows - 1  # 行数
    col = table.ncols - 1  # 列数

    datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
    x_tick = table.col_values(0)[1:]
    x_tick = [f'${e}$' for e in x_tick]
    y_tick = []

    for x in range(1, col+1):
        cols = np.matrix(table.col_values(x)[1:])  # 把list转换为矩阵进行矩阵操作

        y_tick.append(f'${table.col_values(x)[0]}$')
        cols = cols.astype(np.float64)
        datamatrix[:, x-1] = cols  # 按列把数据存进矩阵中

    return x_tick, y_tick, datamatrix


def draw_heatmap(fid):
    xt, yt, corr = excel_to_matrix("../data/corrr.xlsx", fid)
    print(xt)
    mask = np.zeros_like(corr)
    mask[np.tril_indices_from(mask)] = True
    # sns.heatmap(data, cmap='Blues', annot=True)
    fig = plt.figure(figsize=(12, 9))

    clist = ['#E4F9DC', '#0e9647', '#23570F']
    # clist = ['#d8f3dc', '#95d5b2', '#52b788', '#2d6a4f', '#081c15']
    newcmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        'chaos', clist)
    sns.heatmap(corr, cmap=newcmp,  linewidths=0,
                annot=True, fmt='.2f', mask=mask.T, xticklabels=xt, yticklabels=yt)
    plt.savefig(f"../img/new_corr{fid}.pdf")
    plt.show()


draw_heatmap(0)
draw_heatmap(1)
# draw_heatmap(2)
