import numpy as np
import matplotlib.pyplot as plt

#任意计算两点之间距离
def cal_pairwise_dist(data):
	expand_ = data[:, np.newaxis, :]
	repeat1 = np.repeat(expand_, data.shape[0], axis=1)
	repeat2 = np.swapaxes(repeat1, 0, 1)
	D = np.linalg.norm(repeat1 - repeat2, ord=2, axis=-1, keepdims=True).squeeze(-1)
	return D

#计算近邻        
def get_n_neighbors(data, n_neighbors):
	dist = cal_pairwise_dist(data)
	dist[dist < 0] = 0
	n = dist.shape[0]
	N = np.zeros((n, n_neighbors))
	for i in range(n):
		# np.argsort 列表从小到大的索引
		index_ = np.argsort(dist[i])[1:n_neighbors+1]
		N[i] = N[i] + index_
	return N.astype(np.int32)

#LLE过程
def LLE(data, n_dims, n_neighbors,jump=True):
	N = get_n_neighbors(data, n_neighbors)            # k近邻索引
	n, D = data.shape                                 # n_samples, n_features
	# prevent Si to small
	if n_neighbors > D:
		tol = 1e-3
	else:
		tol = 0
	# calculate W
	W = np.zeros((n_neighbors, n))
	I = np.ones((n_neighbors, 1))
	for i in range(n):                                # data[i] => [1, n_features]
		Xi = np.tile(data[i], (n_neighbors, 1)).T     # [n_features, n_neighbors]
		                                              # N[i] => [1, n_neighbors]
		Ni = data[N[i]].T                             # [n_features, n_neighbors]
		Si = np.dot((Xi-Ni).T, (Xi-Ni))               # [n_neighbors, n_neighbors]
		Si = Si + np.eye(n_neighbors)*tol*np.trace(Si)
		Si_inv = np.linalg.pinv(Si)
		wi = (np.dot(Si_inv, I)) / (np.dot(np.dot(I.T, Si_inv), I)[0,0])
		W[:, i] = wi[:,0]
	W_y = np.zeros((n, n))
	for i in range(n):
		index = N[i]
		for j in range(n_neighbors):
			W_y[index[j],i] = W[j,i]
	I_y = np.eye(n)
	M = np.dot((I_y - W_y), (I_y - W_y).T)
	eig_val, eig_vector = np.linalg.eig(M)

	
	if jump==True:
		index_=np.argsort(eig_val)
		eig_val=eig_val[index_]
		j=0
		while eig_val[j]<1e-6:
			j+=1
		index_=index_[j:j+n_dims]
		eig_vec_picked=eig_vector[:,index_]
		Y=eig_vec_picked

	if jump==False:
		index_ = np.argsort(np.abs(eig_val))[1:n_dims+1]
		Y = eig_vector[:, index_]
	return Y
