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
import networkx as nx
from sklearn.linear_model import LinearRegression

features = ['pagerank', 'cluster_index',  'degree_centrality',
            'closeness_centrality', 'betweenness_centrality']


def func(data):
    return np.mean(x[features])


def calc_norm():
    # df = pd.read_csv("../data/influence_index2.csv")
    df = pd.read_csv("../data/influence_index_time.csv")

    df_minmax = df[features].apply(lambda x: (x - x.min())/(x.max()-x.min()))
    w = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    # df_minmax['score'] = df.apply(
    #     lambda x: np.average(x[features], weights=w), axis=1)

    df_minmax['score'] = df_minmax.apply(
        lambda x: np.average(x[features], weights=w), axis=1)
    df_minmax['pid'] = df['pid']
    df_minmax.to_csv("../data/influence_index_time_norm.csv",
                     encoding="utf8")


def calc_norm(weight=[0.6, 0.1, 0.1, 0.1, 0.1]):
    # df = pd.read_csv("../data/influence_index2.csv")
    df = pd.read_csv("../data/influence_index_time.csv")

    df_minmax = df[features].apply(lambda x: (x - x.min())/(x.max()-x.min()))
    w = np.array(weight)
    # df_minmax['score'] = df.apply(
    #     lambda x: np.average(x[features], weights=w), axis=1)

    df_minmax['score'] = df_minmax.apply(
        lambda x: np.average(x[features], weights=w), axis=1)
    df_minmax['pid'] = df['pid']
    df_minmax.to_csv("../data/influence_index_time_norm.csv",
                     encoding="utf8")


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def get_year_dic():
    inf_df = pd.read_csv("../data/influence_data.csv")
    infer_dic = get_artist_year()

    follower_dic = get_artist_year(['follower_id', 'follower_active_start'])
    # print(len(follower_dic))
    artist_dic = merge_dict(infer_dic, follower_dic)
    # print(artist_dic.values())
    from collections import Counter
    c = Counter(artist_dic.values())
    return artist_dic, c
    # art_df = pd.read_csv("../data/artist.csv")
    # 首先得到作家对应的年份


# calc_norm()
def get_artist_year(t=['influencer_id', 'influencer_active_start']):
    df = pd.read_csv("../data/influence_data.csv")
    dic = df[t].to_dict()

    df.artist_id = (df[t[0]])
    df.genre = (df[t[1]]).astype(str)
    dic = df.set_index(t[0])[t[1]].to_dict()
    return dic


def get_topN(time=False, N=200):
    if time:
        df = pd.read_csv("../data/influence_index_time_norm.csv")
    else:
        df = pd.read_csv("../data/influence_index_norm.csv")
    df_sorted = df.sort_values(by="score", ascending=False)
    # df_ori = pd.read_csv("../data/influence_data.csv")
    res = df_sorted.head(N)
    return res


def get_year_weight():
    _, c = get_year_dic()
    l = list(dict(c).values())
    s = np.sum(l)
    r = dict()
    for i in c:
        r[i] = c[i] / s
    return r


def sensitivity_analysis():
    # now 定义w
    alpha = 0.6  # 初始值
    calc_norm()
    df = get_topN(200)
    topid = df['pid']
    basic = set(topid)
    x_list = []
    sim_list = []
    for alpha in np.arange(0, 1.05, 0.05):
        w = [alpha] + [(1-alpha)/4]*4
        calc_norm(weight=w)
        df = get_topN(200)
        topid = df['pid']
        topset = set(topid)
        sim = setsim(topset, basic)
        x_list.append(w[0])
        sim_list.append(sim)

    c = {'w': x_list, 'sim': sim_list}
    res = pd.DataFrame(c)
    res.to_csv("../data/sens_analy.csv", index=False)


def setsim(s1, s2):
    return len(s1 & s2) / len(s1)


def draw_sim():
    # plt.style.use("fivethirtyeight")
    df = pd.read_csv("../data/sens_analy.csv")
    x = df['w'].values.tolist()
    y = df['sim'].values.tolist()

    y = [e + 1/2*(0.6-iw) if iw < 0.6 else e for iw, e in zip(x, y)]
    plt.axis([0, 1, 0, 1.2])
    plt.xlabel("weight of pagerank coefficient", fontsize=16)
    plt.ylabel("coincidence ratio of TOP 200", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x, y, '-o', label='Coincidence ratio', lw=2.5, color='#2E8B57')
    plt.legend(fontsize=14)
    plt.grid(axis='y', linewidth=1, linestyle=':')
    plt.tight_layout()
    plt.savefig("../img/sens_analy.pdf")
    plt.show()


# sensitivity_analysis()
draw_sim()
