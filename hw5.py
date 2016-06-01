# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class SVM(object):
	def __init__(self):

	def linear_kernel(self, x, y):
		return np.dot(x, y.T)

	def poly_kernel(self, x, y, gamma, r, d):
		return (gamma * np.dot(x, y.T) + r) ** d

	def rbf_kernel(self, x, y, sigma):
		return np.linalg.norm(x - y, ord = 2) ** 2 / (-2 * sigma ** 2)

	def exp_kernel(self, x, y, sigma):
		norm = np.zeros((x.shape[0], y.shape[0]))
		for i in range(x.shape[0]):
			for j in range(y.shape[0]):
				norm[i, j] = sum(abs(x[i, :] - y[j, :]))
		return np.exp(-norm * (1 / (2 * sigma **2)))

	def tanh_kernel(self, x, y, beta, b):
		inner = beta * np.dot(x.T, x) + b
		return (np.exp(inner) - np.exp(-inner)) / (np.exp(inner) + np.exp(-inner))

	def svm_train(data, label):


	def svm_classify()
