# # coding:utf-8

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits
# from mpl_toolkits.mplot3d import Axes3D


# def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
#     #Generate a swiss roll dataset.
#     t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
#     x = t * np.cos(t)
#     y = 83 * np.random.rand(1, n_samples)
#     z = t * np.sin(t)
#     X = np.concatenate((x, y, z))
#     X += noise * np.random.randn(3, n_samples)
#     X = X.T
#     t = np.squeeze(t)
#     return X, t

# def rbf(dist, t = 1.0):
#     '''
#     rbf kernel function
#     '''
#     return np.exp(-(dist/t))

# def cal_pairwise_dist(x):

#     '''计算pairwise 距离, x是matrix
#     (a-b)^2 = a^2 + b^2 - 2*a*b
#     '''
#     sum_x = np.sum(np.square(x), 1)
#     dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
#     #返回任意两个点之间距离的平方
#     return dist

# def cal_rbf_dist(data, n_neighbors = 10, t = 1):

#     dist = cal_pairwise_dist(data)
#     dist[dist < 0] = 0
#     n = dist.shape[0]
#     rbf_dist = rbf(dist, t)

#     W = np.zeros((n, n))
#     for i in range(n):
#         index_ = np.argsort(dist[i])[1:1+n_neighbors]
#         W[i, index_] = rbf_dist[i, index_]
#         W[index_, i] = rbf_dist[index_, i]

#     return W

# def le(data,n_dims = 2,n_neighbors = 5, t = 1.0, jump=True):
#     '''

#     :param data: (n_samples, n_features)
#     :param n_dims: target dim
#     :param n_neighbors: k nearest neighbors
#     :param t: a param for rbf
#     :return:
#     '''
#     N = data.shape[0]
#     W = cal_rbf_dist(data, n_neighbors, t)
#     D = np.zeros_like(W)
#     for i in range(N):
#         D[i,i] = np.sum(W[i])

#     D_inv = np.linalg.inv(D)
#     L = D - W
#     eig_val, eig_vec = np.linalg.eig(np.dot(D_inv, L))

#     sort_index_ = np.argsort(eig_val)

#     eig_val = eig_val[sort_index_]
#     # print("eig_val[:10]: ", eig_val[:10])

#     j = 0
#     if jump==True:
#         while eig_val[j] < 1e-6:
#             j+=1
#     # if jump==False:
#     #     j=0

#     # print("j: ", j)

#     sort_index_ = sort_index_[j:j+n_dims]
#     eig_val_picked = eig_val[j:j+n_dims]
#     # print(eig_val_picked)
#     eig_vec_picked = eig_vec[:, sort_index_]

#     # print("L: ")
#     # print(np.dot(np.dot(eig_vec_picked.T, L), eig_vec_picked))
#     # print("D: ")
#     # D not equal I ???
    
#     # print(np.dot(np.dot(eig_vec_picked.T, D), eig_vec_picked))

#     X_ndim = eig_vec_picked
#     return X_ndim




import numpy as np

#热核函数,默认sigama=1
def rbf(dist, t = 1.0):
    return np.exp(-(dist/t))

#计算距离矩阵
def cal_pairwise_dist(x):
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist

# 距离矩阵
# def euclidDistance(x1, x2, sqrt_flag=False):
#     res = np.sum((x1-x2)**2)
#     if sqrt_flag:
#         res = np.sqrt(res)
#     return res
# def cal_pairwise_dist(X):
#     X = np.array(X)
#     S = np.zeros((len(X), len(X)))
#     for i in range(len(X)):
#         for j in range(i+1, len(X)):
#             S[i][j] = 1.0 * euclidDistance(X[i], X[j])
#             S[j][i] = S[i][j]
#     return S

#计算邻接矩阵W：相似度矩阵
def cal_rbf_dist(data, n_neighbors, t = 1):
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)

    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1+n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W

#Laplacian Eigenmaps重构或降维
def le(data,n_dims = 2,n_neighbors = 5, t = 1.0,jump=True):

    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)
    D = np.zeros_like(W)
    for i in range(N):
        D[i,i] = np.sum(W[i])

    #矩阵求逆
    D_inv = np.linalg.inv(D)
    L = D - W
    #计算特征值和特征向量
    eig_val, eig_vec = np.linalg.eig(np.dot(D_inv, L))

    sort_index_ = np.argsort(eig_val)

    eig_val = eig_val[sort_index_]
    
    j = 0
    if jump==True:
        while eig_val[j] < 1e-6:
            j+=1

    sort_index_ = sort_index_[j:j+n_dims]
    eig_val_picked = eig_val[j:j+n_dims]
    # print(eig_val_picked)
    eig_vec_picked = eig_vec[:, sort_index_]

    X_ndim = eig_vec_picked
    return X_ndim
