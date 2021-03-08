

import pickle
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
from scipy import spatial

import matplotlib.pyplot as plt
import networkx as nx


def create_subgraph(G, sub_G, start_node):
    for n in G.neighbors(start_node):
        sub_G.add_path([start_node, n])
        create_subgraph(G, sub_G, n)


# G = nx.DiGraph()
# G.add_path([1, 2, 3, 4])
# G.add_path([3, 'a', 'b'])
# sub_G = nx.DiGraph()
# create_subgraph(G, sub_G, 3)

# G = nx.star_graph(20)
# pos = nx.spring_layout(G)
# colors = range(20)
# options = {
#     "node_color": "#A0CBE2",
#     "edge_color": colors,
#     "width": 4,
#     "edge_cmap": plt.cm.Blues,
#     "with_labels": False,
# }
# nx.draw(G, pos, **options)
# plt.show()

# 来生成一个有10个节点，连接概率为0.6的随机网络

def load_data(name):
    with open(f"../data/{name}.file", "rb") as f:
        d = pickle.load(f)

    return d
