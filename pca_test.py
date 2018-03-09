from numpy import *
from scipy.linalg import *
from pylab import *
import matplotlib.pyplot as plt


def testeig(I, t):  # 提取前t个特征值
    m, n = I.shape
    mean_n = I.mean(axis=0)
    I1 = I - mean_n
    M = (dot(I1.T, I1)) / (m - 1)
    e, EV = eig(M)
    e1 = argsort(e)
    e1 = e1[::-1]
    Q = EV[:, e1[0:t]]
    per = 0
    for i in range(t):
        per += e[e1[i]]
    percent = per / sum(e)
    percent1 = real(percent) * 100
    lowD = dot(I1, Q)  # 降维后的数据
    pj_train = dot(lowD.T, I1)  # 数据投影
    return pj_train, percent1


I = rand(2, 10)
pj_train, percent1 = testeig(I, 2)
plot(I[0, :], I[1, :], 'rs')
X = [I[0, :], pj_train[0, :]]
Y = [I[1, :], pj_train[1, :]]
plot(pj_train[0, :], pj_train[1, :], 'bo')
plot(X, Y)
title('Plotting:PCA')
show()
print(percent1, '\n', pj_train )