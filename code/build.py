import numpy.matlib
import pickle
import json
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

from process_inedx import get_year_dic, get_year_weight, get_artist_year


def test(path="../data/influence_data.csv"):
    WS = nx.random_graphs.watts_strogatz_graph(
        20, 4, 0.3)  # 生成包含20个节点、每个节点4个近邻、随机化重连概率为0.3的小世界网络
    pos = nx.circular_layout(WS)  # 定义一个布局，此处采用了circular布局方式

    colors = range(20)
    options = {
        "node_color": "#A0CBE2",
        "width": 4,
        "edge_cmap": plt.cm.Blues,
        "with_labels": False,
    }
    nx.draw(WS, pos, **options)  # 绘制图形
    plt.show()
    min_p = nx.average_shortest_path_length(WS)
    cluster_index = nx.average_clustering(WS)

    print(f"min dis: {min_p}")
    print(f'cluster index: {cluster_index}')


def basic_index(G):
    def dict2df(data):
        return pd.DataFrame.from_dict(
            data, orient='index')
    # 研究小世界特性时，看成弱连通图
    UG = G.to_undirected()

    all_connected_subgraphs = (UG.subgraph(c).copy()
                               for c in nx.connected_components(UG))

    # 调整权重，
    # print((list(all_connected_subgraphs)))
    BG = list(all_connected_subgraphs)[0]
    print(BG.number_of_nodes())
    # ----------- 聚集系数 -------------------#
    """聚集系数

    在图论中，集聚系数（也称群聚系数、集群系数）是用来描述一个图中的顶点之间结集成团的程度的系数。具体来说，是一个点的邻接点之间相互连接的程度。例如生活社交网络中，你的朋友之间相互认识的程度
    """
    # 平均聚集系数
    cluster_index_mean = nx.average_clustering(BG)
    print(f'average cluster index: {cluster_index_mean}')
    # pagerank 算法
    node_w = get_node_weight(G)
    pr = nx.pagerank(G, alpha=0.85, weight='weight', personalization=node_w)
    df = pd.DataFrame.from_dict(
        pr, orient='index', columns=['pagerank'])

    df_ori = pd.read_csv("../data/influence_data.csv")

    # # 每个节点的聚集系数
    cluster_index_all = nx.clustering(G)
    df['cluster_index'] = dict2df(cluster_index_all)

    # Degree centrality measures.（点度中心性？
    c_degree = nx.degree_centrality(G)
    df['degree_centrality'] = dict2df(c_degree)
    # Closeness centrality measures.（紧密中心性？）
    c_closeness = nx.closeness_centrality(G)
    df['closeness_centrality'] = dict2df(c_closeness)
    # Betweenness centrality measures.（介数中心性？）
    c_betweenness = nx.betweenness_centrality(G, weight='weight')
    df['betweenness_centrality'] = dict2df(c_betweenness)

    df.to_csv("../data/influence_index_time.csv", encoding="utf8")

    # diameter = nx.diameter(BG)
    # print(diameter)

    # for SG in all_connected_subgraphs:
    #     if SG.number_of_nodes() > 2:
    #         # min_p=nx.average_shortest_path_length(SG)
    #         cluster_index = nx.average_clustering(SG)
    #         # print(f"min dis: {min_p}")
    #         print(f'cluster index: {cluster_index}')
    #         print("-" * 100)


def create_subgraph(G, node):
    edges = nx.dfs_successors(G, node)
    nodes = []
    for k, v in edges.items():
        nodes.extend([k])
        nodes.extend(v)
    return G.subgraph(nodes)


def find_subgraph(G, iid=130173, save=False):
    sub_G = create_subgraph(G, iid)
    # print(sub_G.nodes())

    # pos = nx.spring_layout(sub_G)
    if save:
        nx.write_gexf(sub_G, f"../img/subg{iid}.gexf")
    return sub_G


def create_graph(path="../data/influence_data.csv"):
    df = pd.read_csv(path)
    # print(df)
    G = nx.DiGraph()  # 建立一个空的无向图G
    p_infer_list = df['influencer_id'].values.tolist()
    p_follow_list = df['follower_id'].values.tolist()
    infer_genre_list = df['influencer_main_genre']
    follower_genre_list = df['follower_main_genre']

    for p, q, m, n in zip(p_infer_list, p_follow_list, infer_genre_list, follower_genre_list):
        G.add_node(p)
        G.add_node(q)
        G.add_edge(p, q)
        w = 1 if m == n else 0.6

        G.add_weighted_edges_from([(p, q, w)])
    return G


def get_inout_degrees(G):
    ind = G.in_degree()
    outd = G.out_degree()
    od = sorted(dict(outd).values(), reverse=True)
    ind = sorted(dict(ind).values(), reverse=True)
    return ind, od


def build(path="../data/influence_data.csv"):
    G = create_graph()

    # print(nx.degree_histogram(G))
    deg_dist = nx.degree_histogram(G)
    # _, deg_dist = get_inout_degrees(G)

    degree = list(dict(nx.degree(G)).values())
    # degree = deg_dist
    degree = [d for d in degree if d > 0]
    plt.plot(range(0, len(degree)), sorted(degree, reverse=True),
             'ko', color='#40916c', alpha=0.7)
    plt.legend(['degree'])
    plt.xlabel('id')
    plt.ylabel('degree')

    plt.tight_layout()  # 去除pdf周围白边

    plt.savefig("../img/degree_distrubition.pdf")

    # plt.show()
    plt.figure()
    import powerlaw
    fit = powerlaw.Fit(degree, xmin=3, discrete=True)
    print(f'alpha: {fit.power_law.alpha}')
    print(f'x-min: {fit.power_law.xmin}')
    print(f'D: {fit.power_law.D}')
    fit.plot_pdf(color='#2d6a4f', lw=3, linestyle=':', label='curve')

    x, y = powerlaw.pdf(degree, linear_bins=True)

    ind = y > 0
    y = y[ind]
    x = x[:-1]
    x = x[ind]
    plt.scatter(x, y, color='#74c69d', marker='s',
                alpha=0.6, s=50, label='original degree')
    fit.power_law.plot_pdf(color='#1b4332', linestyle='-',
                           label='$\gamma=1.8$', lw=2)
    # fit.power_law.plot_pdf(color='#1b4332', linestyle='-',
    #                        label='$gamma=%.3f$' % fit.power_law.alpha, lw=2)
    plt.legend()
    plt.xlabel('ln(id)')
    plt.ylabel('ln(degree)')

    # plt.title(
    #     'Power Law Curve in Double Logarithmic Coordinates')
    plt.savefig("../img/powerlaw.pdf", dpi=600)
    plt.show()
    # 计算皮尔逊 指数
    # from scipy.stats import pearsonr
    # r, p = pearsonr(x, y)
    # print(f'pearsonr r: {r}, p: {p}')

    # 线性回归
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # print(slope, intercept, r_value, p_value, std_err)

    # 分布
    # DataFitAndVisualization(X, Y)
    # plt.loglog(x, y, color="steelblue", linewidth=2)  # 在双对数坐标轴上绘制度分布曲线

    r1, p1 = fit.distribution_compare('power_law', 'exponential', nested=True)
    r2, p2 = fit.distribution_compare('power_law', 'lognormal', nested=True)
    r3, p3 = fit.distribution_compare(
        'power_law', 'truncated_power_law', nested=True)
    r4, p4 = fit.distribution_compare(
        'power_law', 'stretched_exponential', nested=True)
    print(r1, p1)
    print(r2, p2)
    print(r3, p3)
    print(r4, p4)


def DataGenerate():
    X = np.arange(10, 1010, 10)  # 0-1，每隔着0.02一个数据 0处取对数,会时负无穷 生成100个数据点
    noise = norm.rvs(0, size=100, scale=0.2)  # 生成50个正态分布 scale=0.1控制噪声强度
    Y = []
    for i in range(len(X)):
        Y.append(10.8*pow(X[i], -0.3)+noise[i])  # 得到Y=10.8*x^-0.3+noise

    # plot raw data
    Y = np.array(Y)
    plt.title("Raw data")
    plt.scatter(X, Y, color='black')
    plt.show()

    X = np.log10(X)  # 对X，Y取双对数
    Y = np.log10(Y)
    return X, Y


def DataFitAndVisualization(X, Y):
 # 模型数据准备
    X_parameter = []
    Y_parameter = []
    for single_square_feet, single_price_value in zip(X, Y):
        X_parameter.append([float(single_square_feet)])
        Y_parameter.append(float(single_price_value))

    # 模型拟合
    regr = linear_model.LinearRegression()
    regr.fit(X_parameter, Y_parameter)
    # 模型结果与得分
    print('Coefficients: \n', regr.coef_,)
    print("Intercept:\n", regr.intercept_)
    # The mean square error
    print("Residual sum of squares: %.8f" %
          np.mean((regr.predict(X_parameter) - Y_parameter) ** 2))  # 残差平方和

    # 可视化
    plt.title("Log Data")
    plt.scatter(X_parameter, Y_parameter, color='black')
    plt.plot(X_parameter, regr.predict(X_parameter), color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())
    plt.show()

    # plt.loglog(x[1:], y[1:], color="steelblue", linewidth=1)  # 在双对数坐标轴上绘制度分布曲线
    # plt.show()

    # nx.write_gexf(G, "../img/test.gexf")
    # nx.draw(G)  # 绘制网络G
    # plt.savefig("../img/ba.png")  # 输出方式1: 将图像存为一个png格式的图片文件
    # plt.show()  # 输出方式2: 在窗口中显示这幅图像


def show_distribution(G, show=False):
    ind = G.in_degree()
    outd = G.out_degree()
    deg_dist = sorted(dict(outd).values(), reverse=True)
    # print(deg_dist)
    #  = nx.degree_histogram(G)
    plt.loglog(range(0, len(deg_dist)), deg_dist,
               'ko', mec='w', ms=8, color='steelblue')
    plt.xlabel('id')
    plt.ylabel('degree')
    if show:
        plt.show()


def set_weight(G):
    ind = G.in_degree()
    outd = G.out_degree()

    # 遍历所有边
    d, _ = get_year_dic()
    year_w = get_year_weight()
    for (u, v, wt) in G.edges.data('weight'):
        i = G.in_degree[v]
        G[u][v]['weight'] /= ((math.log(i) + 1)+1) * year_w[d[v]]

        # if G[u][v]['weight'] < 0.6:
        #     print('way2-(%d, %d, %.3f)' % (u, v, G[u][v]['weight']))


def update_weight(G):
    ind = G.in_degree()
    outd = G.out_degree()
    # 遍历所有边
    for (u, v, wt) in G.edges.data('weight'):
        i = G.in_degree[v]
        G[u][v]['weight'] = 1 / G[u][v]['weight']
        # if G[u][v]['weight'] < 0.6:
        #     print('way2-(%d, %d, %.3f)' % (u, v, G[u][v]['weight']))


def get_node_weight(G):
    ind = dict(G.in_degree())
    outd = dict(G.out_degree())
    df = pd.DataFrame.from_dict(outd, orient="index", columns=["degree"])
    df_minmax = df.apply(lambda x: (x - x.min())/(x.max()-x.min()))
    sumv = df_minmax['degree'].sum()

    df_weight = df_minmax.apply(lambda x: x / sumv)
    dic = df_weight.to_dict()['degree']
    return dic


def calc_shortest_path(Ga):
    G = Ga.copy()
    update_weight(G)

    f1 = open("../data/length.file", "wb")
    f2 = open("../data/path.file", "wb")

    def dict2df(data):
        return pd.DataFrame.from_dict(
            data, orient='index')
    length = dict(nx.all_pairs_bellman_ford_path_length(G))
    path = dict(nx.all_pairs_bellman_ford_path(G))

    pickle.dump(length, f1)
    pickle.dump(path, f2)
    # print(G[25462])
    # print(nx.has_path(G, 25462, 335))


def save_data(data, name):
    with open(f"../data/{name}.file", "wb") as f:
        pickle.dump(data, f)


def load_data(name):
    with open(f"../data/{name}.file", "rb") as f:
        d = pickle.load(f)

    return d


def get_id():
    df = pd.read_csv("../data/influence_index_norm.csv")
    all_id = df['pid']
    return all_id.tolist()


def calc_influence(G):
    # 现在已经有了最短路径
    paths = load_data("path")
    print("load finish")
    df = pd.read_csv("../data/influence_index_norm.csv")
    all_id = df['pid'].tolist()
    imps_df = df['score']
    imps = {}

    # 遍历所有的节点对
    import itertools

    id_pairs = itertools.product(all_id, all_id)
    dic = {}

    for idx, i in enumerate(all_id):
        imps[i] = imps_df[idx]
    cnt = 0
    for u, v in id_pairs:
        cnt += 1
        if cnt % 100000 == 0:
            print(f"process {cnt}")
        # 对节点u和v计算如下
        # u = 759491
        # v = 913210
        basic = 1
        imp = imps[u]
        try:
            path = paths[u][v]
            for i, j in zip(path[:-1], path[1:]):
                basic *= G[i][j]['weight']
        except KeyError:
            imp = 0
        if u not in dic.keys():
            dic[u] = {}
        dic[u][v] = imp * basic

    save_data(dic, "inf_single2")


def get_genre_dic():
    df = pd.read_csv("../data/artist.csv")
    dic = df[['artist_id', 'genre']].to_dict()

    df.artist_id = (df['artist_id'])
    df.genre = (df['genre']).astype(str)
    dic = df.set_index('artist_id')['genre'].to_dict()
    return dic


def calc_range_influence(G):
    # single2 是没有限制深度的
    # 这就得到了一个流派的影响力之和
    dic = get_genre_dic()

    df = pd.read_csv("../data/artist.csv")
    genre_id_list = df.groupby('genre')['artist_id']
    infd = load_data("inf_single2")

    # 这就得到了一个人受到流派的影响
    be_inf_dic = {}
    for elem in genre_id_list:
        l = list(elem)
        name = l[0]
        grouped = list(l[1])
        q = 0
        for j in grouped:
            tmp = 0
            for k in grouped:
                q += infd[j][k]
                tmp += infd[k][j]
            be_inf_dic[j] = tmp
    inf_df = pd.DataFrame.from_dict(be_inf_dic, orient='index')

    inf_df.to_csv("../data/be_infed.csv")


if __name__ == '__main__':
    # build()
    # test()

    G = create_graph()

    set_weight(G)

    # 为节点的权重赋值
    # get_node_weight(G)
    # show_distribution(G, True)
    # basic_index(G)
    # calc_shortest_path(G)

    # calc_influence(G)
    # calc_range_influence(G)
    SG = find_subgraph(G, 66915, False)

    # cluster_index = nx.average_clustering(SG)
    # print(cluster_index)
    # paths = load_data("path")
    # print(paths[130173][775877])
    # print(G.in_degree[130173])
    # 775877
