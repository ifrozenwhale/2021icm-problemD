from matplotlib.backends.backend_pdf import PdfPages
import ast
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
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
# features = ['Nor_danceability', 'Nor_energy', 'Nor_valence', 'Nor_tempo', 'Nor_loudness',
#             'Nor_acousticness', 'Nor_instrumentalness', 'Nor_liveness', 'Nor_speechiness', 'Nor_duration_ms']
features = ['Nor_danceability', 'Nor_energy', 'Nor_valence', 'Nor_tempo', 'Nor_loudness', 'Nor_key', 'Nor_acousticness',
            'Nor_instrumentalness', 'Nor_liveness', 'Nor_speechiness', 'Nor_duration_ms']

ori_features = ['danceability',    'energy',   'valence',      'tempo',   'loudness',  'key',  'acousticness',
                'instrumentalness',  'liveness',  'speechiness',  'duration_ms']


def distance_euclidean_scipy(vec1, vec2, distance="euclidean"):
    return spatial.distance.cdist(vec1, vec2, distance)


def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


# def cosine_similarity(x, y, norm=False):
#     """ 计算两个向量x和y的余弦相似度 """
#     # print('-' * 100)
#     print(x)
#     print(y)
#     print("-"*1000)
#     assert len(x) == len(y), "len(x) != len(y)"

#     res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]]
#                     for i in range(len(x))])
#     cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

#     return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def calc_except_list(df, gid):
    return df[df['genre_id'] != gid][features].mean().values


def calc(path="../data/artist.csv"):
    df = pd.read_csv(path)
    grouped_genre = df.groupby('genre_id')

    except_list = []
    for i in range(1, 21):
        except_list.append(df[df['genre_id'] != i][features].mean().values)
    center_list = list(grouped_genre[features])
    print(center_list)
    center_list = grouped_genre[features].mean().values
    print(center_list)

    df['dis_in'] = df.apply(lambda x: distance_euclidean_scipy(np.array(list(x[features])).reshape(
        1, -1), np.array(center_list[x['genre_id']-1]).reshape(1, -1))[0, 0], axis=1)
    df['dis_out'] = df.apply(lambda x: distance_euclidean_scipy(
        np.array(list(x[features])).reshape(1, -1), np.array(except_list[x['genre_id']-1]).reshape(1, -1))[0, 0], axis=1)

    df['cos_in'] = df.apply(lambda x: cosine_similarity(
        list(x[features]), center_list[x['genre_id']-1]), axis=1)
    df['cos_out'] = df.apply(lambda x: cosine_similarity(
        (list(x[features])), except_list[x['genre_id']-1]), axis=1)

    # df['cos_in'] = df.apply(lambda x: cosine_similarity(
    #     np.array(list(x[features])).reshape(1, -1), np.array(center_list[x['genre_id']-1]).reshape(1, -1))[0, 0], axis=1)
    # df['cos_out'] = df.apply(lambda x: cosine_similarity(
    #     np.array(list(x[features])).reshape(1, -1), np.array(except_list[x['genre_id']-1]).reshape(1, -1))[0, 0], axis=1)

    df.to_csv('../data/artist_res.csv', encoding='utf8', index=False)


def calc_weight():
    df = pd.read_csv("../data/full_music_data_norm.csv")

    grouped_pm = df.groupby([df['artist_id'], df['year']])

    feature_vars = grouped_pm[ori_features].std()
    feature_vars = feature_vars[feature_vars[ori_features] > 0]
    # feature_vars = feature_vars.drop(['Unnamed: 0'], axis=1)
    # feature_vars = feature_vars.drop(['Unnamed: 0.1'], axis=1)
    res = feature_vars.dropna(axis=0, how="any")
    m = res.mean()
    return m

    # 终于得到了处理后的结果


def pre_norm():
    df = pd.read_csv("../data/full_music_data_res.csv")
    df[ori_features] = df[ori_features].apply(
        lambda x: (x - x.min())/(x.max()-x.min()))

    df.to_csv("../data/full_music_data_norm.csv")


def split_author():
    rows_list = []
    df = pd.read_csv("../data/full_music_data.csv")
    for i in range(len(df)):
        id_list = ast.literal_eval(df.iloc[i]['artists_id'])
        name_list = ast.literal_eval(df.iloc[i]['artist_names'])
        if len(id_list) >= 2:
            for j in range(len(id_list)):
                elem = df.iloc[i].copy()
                elem['artists_id'] = id_list[j]
                elem['artist_names'] = name_list[j]
                # print(id_list[j])
                rows_list.append(elem)
        else:
            elem = df.iloc[i].copy()
            elem['artists_id'] = id_list[0]
            elem['artist_names'] = name_list[0]
            rows_list.append(elem)

            # print(df.iloc[i])
            # 拆分成多个行
    # print(rows_list)
    # print(len(rows_list))
    sdf = pd.DataFrame(rows_list)
    sdf.to_csv("../data/full_music_data_res.csv")


def calc_genre_features():
    df = pd.read_csv("../data/artist.csv")

    d = df.groupby(['genre'], as_index=False).apply(
        lambda x: np.average(x['Nor_key'], weights=x['count']))
    d.to_csv("../data/features_genre.csv")

    # grouped_pm = df.groupby(['genre'])
    # ls_pm = list(grouped_pm)
    # d = ls_pm[0][1]
    # print(d)
    # rs = d.apply(lambda x: np.average(
    #     x['Nor_danceability'], weights=x['count']))

    # print(rs)


def add_genre_data():
    df_all = pd.read_csv("../data/full_music_data_norm.csv")
    df_inf = pd.read_csv("../data/artist.csv")
    df_inf = df_inf[['artist_id', 'genre']]
    result = pd.merge(df_all, df_inf, on=['artist_id'])

    result.to_csv("../data/full_music_data_norm_genre.csv")


def mean(d, w):
    # print(np.array(d))
    return np.average(np.array(d), axis=0, weights=w)


def calculate(x):
    m = mean(x[ori_features], x['weight'])
    # print(list(x['popularity']))
    # print('-' * 100)
    num = len(x['weight'])

    m = np.append(m, num)
    idx = ori_features.copy()
    idx.append('count')
    return pd.Series(m, index=idx)


def calc_range_features():
    df = pd.read_csv("../data/full_music_data_norm_genre.csv")
    df['weight'] = df['popularity'].apply(lambda x: x+1)

    # tmp = df.groupby([df['influencer_id'], df['year']])
    # df.groupby(pd.TimeGrouper('A')).apply(calculate).to_period('A')
    pm = df.groupby([df['influencer_main_genre'], df['year']],
                    as_index=False).apply(calculate)
    pm.to_csv("../data/range_features.csv", index=False)
    # print(grouped_pm)


def minmax_norm():
    df = pd.read_csv("../data/data_by_year.csv")
    df[ori_features] = df[ori_features].apply(
        lambda x: (x - x.min())/(x.max()-x.min()))
    df.to_csv("../data/data_by_year_norm.csv")


def calc_time_sim():
    df = pd.read_csv("../data/data_by_year_norm.csv")
    dis_list = []
    cos_list = []
    year_list = []
    for i in range(1, len(df)):
        line0 = list(df.iloc[i-1][ori_features])
        line1 = list(df.iloc[i][ori_features])
        w = calc_weight()
        s = np.sum((1/w)**2)
        line0_w = [a / b for a, b in zip(line0, w)]
        line1_w = [a / b for a, b in zip(line1, w)]

        year = df.iloc[i]['year']
        year_list.append(year)
        dis_sim = distance_euclidean_scipy(np.array(line0_w).reshape(
            1, -1), np.array(line1_w).reshape(1, -1))[0, 0]
        cos_sim = cosine_similarity(np.array(line0).reshape(
            1, -1), np.array(line1).reshape(1, -1))[0, 0]
        dis_list.append(dis_sim)
        cos_list.append(cos_sim)
    dis_list = [1 - e / np.sqrt(s) for e in dis_list]
    c = {'year': year_list, 'dis': dis_list, 'cos': cos_list}
    res = pd.DataFrame(c)

    res.to_csv("../data/time_sim.csv", index=False)
    # df.apply(lambda x: distance_euclidean_scipy(np.array(list(x[features])).reshape(
    #     1, -1), np.array(center_list[x['genre_id']-1]).reshape(1, -1))[0, 0], axis=1)


def calc_line_sim(l0, l1, alpha=0.3):
    dis_sim = distance_euclidean_scipy(np.array(l0).reshape(
        1, -1), np.array(l1).reshape(1, -1))[0, 0]
    cos_sim = cosine_similarity(np.array(l0).reshape(
        1, -1), np.array(l1).reshape(1, -1))[0, 0]
    # 现在计算加权后的

    dis_sim_new = 1 - dis_sim / np.sqrt(len(features))

    sim = alpha * cos_sim + (1-alpha) * dis_sim_new
    return sim


def calc_genre_time_sim():
    df = pd.read_csv("../data/range_features.csv")
    grouped_year = df.groupby("year")
    res_list = []
    for group in grouped_year:
        # 单独计算某一年
        l = list(group)
        year = l[0]
        data = l[1]
        # print(year)
        # print(data)
        # print("-"*100)
        # 研究 g_name 和其他人的影响

        for i in range(0, len(data)):
            base_line = data.iloc[i][ori_features]
            idx_sim = 0
            max_sim = 0

            for j in range(0, len(data)):
                if i != j:
                    line = data.iloc[j][ori_features]
                    dis_sim = distance_euclidean_scipy(np.array(base_line).reshape(
                        1, -1), np.array(line).reshape(1, -1))[0, 0]
                    cos_sim = cosine_similarity(np.array(base_line).reshape(
                        1, -1), np.array(line).reshape(1, -1))[0, 0]
                    # 现在计算加权后的
                    alpha = 0.3
                    dis_sim_new = 1 - dis_sim / np.sqrt(len(features))

                    sim = alpha * cos_sim + (1-alpha) * dis_sim_new
                    if sim > max_sim:
                        max_sim = sim
                        idx_sim = j

            # i 和 j 最为相似，相似度 max_sim
            name0 = data.iloc[i]['influencer_main_genre']
            name1 = data.iloc[idx_sim]['influencer_main_genre']
            if len(data) > 1:
                res_list.append([year, name0, name1, max_sim])
    res_df = pd.DataFrame(
        res_list, columns=['year', 'genre0', 'genre1', 'sim'])
    print(res_df)
    res_df.to_csv("../data/genre_time_sim.csv", index=False)


def calc1946():
    df = pd.read_csv("../data/Modified_inf.csv")
    year_df = pd.read_csv("../data/data_by_year_norm.csv")
    basic_line = (year_df.loc[year_df['year'] == 1946]
                  [ori_features]).values.tolist()
    res_list = []
    for i in range(len(df)):
        name = df.iloc[i]['artist_name']
        aid = df.iloc[i]['artist_id']
        line = df.iloc[i][features].values.tolist()
        sim = calc_line_sim(line, basic_line)
        res_list.append([aid, name, sim])
    res_df = pd.DataFrame(
        res_list, columns=['artist_id', 'artist_name', 'sim'])
    # print(res_df)
    res_df.to_csv("../data/sim1946.csv", index=False)


def calc_dif_features():
    df = pd.read_csv("../data/range_features.csv")
    year_df = pd.read_csv("../data/data_by_year_norm.csv")
    main_dif_dic = {}
    # 首先计算外部影响

    last_main_year = df.iloc[0]['year']

    for i in range(1, len(year_df)):
        year = (year_df.iloc[i]['year'])
        # 2000 2002 2004 2006
        year = int(year)
        data0 = year_df.iloc[i-1][ori_features].values.tolist()
        data1 = year_df.iloc[i][ori_features].values.tolist()
        dif = [b - a for a, b in zip(data0, data1)]
        main_dif_dic[year] = dif
        last_main_year = year

    # 然后计算内部影响
    grouped_genre = df.groupby("influencer_main_genre")
    all_genre_dic = {}
    for group in grouped_genre:
        # 研究某个group
        l = list(group)
        name = l[0]
        data = l[1]
        single_genre_list = []
        last_year = data.iloc[0]['year']

        for k in range(1, len(data)):
            year = data.iloc[k]['year']
            data0 = data.iloc[k-1][ori_features].values.tolist()
            data1 = data.iloc[k][ori_features].values.tolist()
            dif = [b - a for a, b in zip(data0, data1)]

            single_genre_list.append((year, dif))
            last_year = year
        all_genre_dic[name] = single_genre_list

    # 然后计算差
    # 假设研究的是Pop
    res_dic = {}
    for item in all_genre_dic.items():
        name = item[0]

        tmp = item[1]
        # 这里是一个流派的
        genre_dic = {}
        # print(tmp)
        # print("-"*200)
        for elem in tmp:
            # 这里是各个年的
            year = elem[0]
            data = elem[1]
            # print(data)
            # br()

            # 需要寻找这一年和之前的年的差距
            # base

            basic_data = main_dif_dic[year]
            dif = [a - b for a, b in zip(data, basic_data)]
            if 'Pop' in name:
                # print(year, dif)
                pass
            genre_dic[year] = [data, dif, basic_data]
        res_dic[name] = genre_dic

    draw_dif(fulldata=res_dic, genre='Jazz', feature_ids=[9], color=blue)
    draw_dif(fulldata=res_dic, genre='Jazz', feature_ids=[10], color=green)

# speech
# loudness
# tempo
# acc


# ori_features = ['danceability',    'energy',   'valence',
#                   'tempo',   'loudness',  'key',
#               'acousticness',  'instrumentalness',  'liveness',
    #  'speechiness',                'duration_ms']
# loud acous live speech du
# 9 4 3 6

blue = ['#006382', '#7AB9CC']
green = ['#1F8A70', '#90D1C1']
orange = ['#FD7400', '#FFC999']
red = ['#B22222', '#F08080']


def draw_dif(fulldata, genre, feature_ids=[9], color=green):
    plt.style.use("seaborn-colorblind")
    data = fulldata[genre]
    years = list(data.keys())
    minx, maxx = np.min(years), np.max(years)
    x_year = list(range(minx, maxx+1))
    y_data = [[0]*11] * len(x_year)
    y_data_ori = [[0]*11] * len(x_year)
    y_data_main = y_data.copy()
    for year, d in data.items():
        idx = x_year.index(year)
        y_data_ori[x_year.index(year)] = d[0]
        y_data[x_year.index(year)] = d[1]
        y_data_main[x_year.index(year)] = d[2]

    new_y_data_ori = [[i[k] for k in feature_ids] for i in y_data_ori]
    new_y_data = [[i[k] for k in feature_ids] for i in y_data]
    new_y_data_main = [[i[k] for k in feature_ids] for i in y_data_main]

    plt.plot(x_year, new_y_data_ori, color=color[0], linewidth=2)
    plt.plot(x_year, new_y_data, '--', color=color[1], linewidth=2)
    # plt.plot(x_year, new_y_data_main, '-o', color='#aaaaaa', linewidth=1.5)

    labels0 = [f'{ori_features[i]}(adjusted)' for i in feature_ids]
    labels1 = [ori_features[i] for i in feature_ids]
    # labels2 = [f'{ori_features[i]}(main)' for i in feature_ids]
    plt.xlabel("year")

    plt.legend([labels0[0], labels1[0]])
    plt.tight_layout()  # 去除pdf周围白边
    plt.savefig(f"../img/{genre}_dif_{labels0[0]}.pdf")

    plt.show()


def br():
    print('-'*200)


def setsim(s1, s2):
    return len(s1 & s2) / len(s1)


def sensitivity_analysis():
    # now 定义w
    alpha = 0.3
    df = pd.read_csv("../data/artist.csv")
    # 甲壳虫乐队 754032
    base_line = df[df['artist_id'] == 754032][features].values.tolist()
    print(base_line)
    base_dic = {}
    for i in range(len(df)):
        artist_id = df.iloc[i]['artist_id']
        line = df.iloc[i][features].values.tolist()
        sim = calc_line_sim(base_line, line, alpha)
        base_dic[artist_id] = sim
    base_df = pd.DataFrame.from_dict(base_dic, orient='index', columns=["sim"])
    # print(base_df.sort_values(by="sim", ascending=False))
    top_basic = set(base_df.sort_values(
        by='sim', ascending=False).head(200).index)

    # 计算各个艺术家和它的
    x_list = []
    sim_list = []
    for alpha in np.arange(0, 1.05, 0.05):
        all_dic = {}
        for i in range(len(df)):
            artist_id = df.iloc[i]['artist_id']
            line = df.iloc[i][features].values.tolist()
            sim = calc_line_sim(base_line, line, alpha)
            all_dic[artist_id] = sim
        all_df = pd.DataFrame.from_dict(
            all_dic, orient='index', columns=["sim"])
        print(all_df.sort_values(by="sim", ascending=False).head(50))
        br()
        top_all = set(all_df.sort_values(
            by='sim', ascending=False).head(200).index)

        sim = setsim(top_basic, top_all)
        x_list.append(alpha)
        sim_list.append(sim)

    c = {'w': x_list, 'sim': sim_list}
    res = pd.DataFrame(c)
    res.to_csv("../data/sens_analy_sim.csv", index=False)


def draw_sim2():
    # plt.style.use("fivethirtyeight")
    df = pd.read_csv("../data/sens_analy_sim.csv")
    x = df['w'].values.tolist()
    y = df['sim'].values.tolist()

    # y = [e + 1/2*(0.6-iw) if iw < 0.6 else e for iw, e in zip(x, y)]
    # plt.axis([0, 1, 0, 1.2])

    plt.xlabel("weight of cosine similarity", fontsize=14)
    plt.ylabel("coincidence ratio of TOP 200", fontsize=14)

    plt.plot(x, y, '-o', label='Coincidence ratio', lw=2.5, color='#2E8B57')
    plt.xticks(fontsize=14)
    plt.yticks(np.arange(0, 1.2, 0.1), fontsize=14)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(axis='y', linewidth=1, linestyle=':')
    plt.tight_layout()
    plt.savefig("../img/sens_analy_sim.pdf")
    plt.show()


def draw_sim():
    # plt.style.use("fivethirtyeight")
    df = pd.read_csv("../data/sens_analy.csv")
    x = df['w'].values.tolist()
    y = df['sim'].values.tolist()

    y = [e + 1/2*(0.6-iw) if iw < 0.6 else e for iw, e in zip(x, y)]
    # plt.axis([0, 1, 0, 1.2])

    plt.xlabel("weight of pagerank coefficient", fontsize=14)
    plt.ylabel("coincidence ratio of TOP 200", fontsize=14)

    plt.plot(x, y, '-o', label='Coincidence ratio', lw=2.5, color='#2E8B57')
    plt.xticks(fontsize=14)
    plt.yticks(np.arange(0, 1.2, 0.1), fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linewidth=1, linestyle=':')
    plt.tight_layout()
    plt.savefig("../img/sens_analy.pdf")
    plt.show()


if __name__ == '__main__':
    # a = [[1, 2, 3, 4, 5, 7], [9, 8, 7, 6, 5, 4], [1, 2, 3, 2, 1, 2]]
    # print(except_list(a, 2))
    # calc()
    # split_author()
    # pre_norm()
    # calc_weight()
    # calc_genre_features()

    # add_genre_data()
    # calc_range_features()

    # minmax_norm()
    # calc_time_sim()

    # calc_genre_time_sim()
    # calc1946()
    # calc_dif_features()

    # sensitivity_analysis()
    draw_sim()
