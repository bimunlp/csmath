# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 

def Jacobi(a, b, x):
	dim = len(x)
	J = np.zeros((dim, 2))
	for i in range(dim):
		naka = np.exp(-b * x[i])
		J[i, :] = np.mat((naka, - a * x[i] * naka)) 
	return J

def LM(x, y, eps = 10**(-15), maxIts = 100):
	a = 10
	b = 0.5

	it = 0
	lmd = 0.01
	v = 2
	dim = len(x)

	updataJ = 1

	for i in range(maxIts):
		if updataJ == 1:
			y_est = a * np.exp(-b * x)
			d = (y - y_est).T
			J = Jacobi(a, b, x)
			H = np.dot(J.T, J)

			if i == 0:
				e = np.dot(d.T, d)

		H_lm = H + lmd * np.eye(v)
		dp = np.dot(np.dot(np.linalg.inv(H_lm), J.T), d)

		g = np.dot(J.T, d)
		a_lm = a + dp[0]
		b_lm = b + dp[1]

		y_est_lm = a_lm * np.exp(-b_lm * x)
		d_lm = (y - y_est_lm).T
		e_lm = np.dot(d_lm.T, d_lm)

		if e_lm < e:
			lmd = lmd/10
			a = a_lm
			b = b_lm
			e = e_lm
			updataJ = 1
		else:
			updataJ = 0
			lmd = lmd * 10

		if e < eps:
			break

	return a, b

def fig(x, y, a, b):
	x_show = np.linspace(0, 10, 1000)
	y_show = []
	for xi in x_show:
		y_show.append(a * np.exp(-b * xi))

	plt.plot(x_show, y_show)
	plt.plot(x, y, 'o')
	plt.show()

if __name__ == '__main__':
	x_data = np.linspace(0, 10, 10)
	y0 = 7 * np.exp(-5 * x_data )
	y_data = [np.random.normal(0, 0.15) + yi for yi in y0]
	a, b = LM(x_data, y_data)
	print "a = %s" %a
	print "b = %s" %b

	fig(x_data, y_data, a, b)
	