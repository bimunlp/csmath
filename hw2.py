# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name, num):
	final_list = []
	file_data = open(file_name)
	for line in file_data:
		l_list = map(int, line.strip('\n').split(','))
		if l_list[-1] == num:
			final_list.append(l_list)
	file_data.close()
	return np.mat(final_list)[:, :-1]

def pca(X, pl):
	X_mean = np.mean(X, axis = 0)
	new_X = X - X_mean
	X_cov = np.cov(new_X, rowvar = 0)
	U, S, V = np.linalg.svd(X_cov)
	u = U[:, 0:pl]
	v = V[:pl, :]

	lowD = new_X * u
	reconD = lowD * u.T + X_mean

	return X_mean, lowD, v

def select_point(feature):
	grid = []
	for i in range(-2, 3):
		for j in range(-2, 3):
			grid.append([i*7, j*7])

	points_ind = []
	for gd in np.mat(grid):
		points_ind.append(np.argmin(np.linalg.norm(feature - gd, axis = 1)))
	points = feature[points_ind, :]
	return points


def plot(feature, points, num):
	plt.title('PCA result, digit %d' %num)
	plt.plot(feature[:, 0], feature[:, 1], 'g.', points[:, 0], points[:, 1], 'ro')
	plt.xlabel('first principal component')
	plt.ylabel('second principal compenent')
	plt.show()

def fig(X_mean, points, v):
	img = np.zeros((8*5, 8*5))
	for i in range(5):
		for j in range(5):
			re_X = X_mean + points[5*i+j, :] * v
			img[i*8:(i+1)*8, j*8:(j+1)*8] = re_X.reshape((8, 8))
	plt.title('Result')
	plt.imshow(img, cmap = plt.cm.gray)
	plt.show()


if __name__ == '__main__':
	test = read_data('optdigits.tra', 3)
	mean, low, v = pca(test, 2)
	points = select_point(low)
	plot(low, points, 3)
	fig(mean, points, v)


