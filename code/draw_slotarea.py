import matplotlib.pyplot as plt
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


def load_data():
    df = pd.read_csv("../data/range_features.csv")
    grouped_genre = df.groupby('influencer_main_genre')
    min_year = df['year'].min()
    max_year = df['year'].max()

    year_list = list(range(min_year, max_year+1))
    # print(year_list)
    all_data_list = []
    labels = []
    for group in grouped_genre:
        yg_list = [0] * len(year_list)
        l = list(group)
        g = l[0]
        labels.append(g)
        year = list(l[1]['year'])
        count = list(l[1]['count'])
        for yid, y in enumerate(year):
            if y < 1922:
                continue
            idx = year_list.index(y)
            yg_list[idx] = count[yid]
        all_data_list.append(yg_list)
    # 现在处理百分比
    arr = np.array(all_data_list)
    # A = arr / arr.sum(axis=0)[np.newaxis, :]
    # A = np.nan_to_num(A)
    # print(A)
    return year_list, arr, labels


def draw():
    # 使用fivethirtyeight这个超漂亮的风格
    fig = plt.figure(figsize=(20, 8))

    # plt.style.use("fivethirtyeight")
    ax = plt.axes()
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        spine.set_visible(False)
    # 初始化横坐标的所有值(这里表示为时间的变化)
    minutes, data, labels = load_data()
    # 初始化所有不同数据集纵坐标表示的值(可以表示团队个人一天工作时间的分配)
    # player1 = [8, 6, 5, 5, 4, 2, 1, 1, 0]
    # player2 = [0, 1, 2, 2, 2, 4, 4, 4, 4]
    # player3 = [0, 1, 1, 1, 2, 2, 3, 3, 4]

    # labels = ['player1', 'player2', 'player3']
    # colors = ['#6d904f', '#fc4f30', '#008fd5']
    colors = [
        '#51574a',  '#447c69',   '#74c493',
        '#8e8c6d',  '#e4bf80',  '#e9d78e',
        '#e2975d',  '#f19670',  '#e16552',
        '#c94a53',  '#be5168',  '#a34974',
        '#993767',  '#65387d',   '#4e2472',
        '#9163b6',  '#e279a3',  '#e0598b',
        '#7c9fb0',  '#5698c4',  '#9abf88'
    ]
    # colors = ['#eb7faf', '#923a60', '#51001b',
    #           '#6200ea', '#2962ff', '#00bfa5',
    #           '#0091ea', '#00b8d4', '#00c853',
    #           '#00c853', '#00c853', '#00c853',
    #           '#00c853',  '#304ffe', '#ff6d00',
    #           '#dd2c00', '#dd2c00', '#dd2c00',
    #           '#dd2c00']
    # stackplot的特点就是可易很方便的看出不同数据集之间每一个特定点的差异
    # 注意传入的多个可迭代对象的维度应该相同
    ax = plt.axes()

    plt.stackplot(minutes, data, labels=labels, colors=colors)
    # legend接受loc参数可以改变显示标签的放置位置, 可以用一个元组加两个数来表示距离坐标轴原点的百分比距离,\
    #  也可以使用字符串表示:
    '''
        best
        upper right
        upper left
        lower left
        lower right
        right
        center left
        center right
        lower center
        upper center
        center
        例: plt.legend(loc='upper left')
    '''
    plt.legend(loc='center right', fontsize=14)
    plt.grid(linewidth=0)
    # plt.title("First Stack Plot
    # 美化输出

    plt.yticks([])
    plt.tight_layout()
    plt.savefig("../img/area_ori.pdf")
    plt.show()


# load_data()
draw()
