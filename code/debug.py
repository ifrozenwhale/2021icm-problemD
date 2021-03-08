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

from sklearn.metrics.pairwise import cosine_similarity, paired_distances

import numpy as np
from scipy import interpolate
import pylab as pl

x = np.linspace(0, 10, 10)
x = list(range(0, 10))
y = np.sin(x)

xnew = [0, 0, 0, 0, 1, 4, 2, 2, 5, 6, 5]

pl.plot(x, y, 'ro')
list1 = ['linear', 'nearest']
list2 = [0, 1, 2, 3]
for kind in list1:
    print(kind)
    f = interpolate.interp1d(x, y, kind=kind)
    # f是一个函数，用这个函数就可以找插值点的函数值了：
    ynew = f(xnew)
    pl.plot(xnew, ynew, label=kind)

pl.legend(loc='lower right')
pl.show()
