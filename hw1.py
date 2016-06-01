# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 

class CurveFitting(object):

	def __init__(self, n, degree, reg, ex):
		self.n = n
		self.degree = degree + 1
		self.reg = reg
		self.ex = ex
		self.x = np.linspace(0, 1, self.n)
		self.y = np.zeros((self.n, 1))
		self.w = np.zeros((degree, 1))
		
	def generate_sin(self):
		y0 = np.sin(self.x * 2 * np.pi)
		self.y = [np.random.normal(0, 0.15) + yi for yi in y0]

	def poly_fit(self):
		if self.reg:
			vreg = np.eye(self.degree) * np.exp(self.ex)
		else:
			vreg = np.zeros((self.degree, self.degree))

		vx = np.ones((self.n, self.degree))
		for i in range(self.n):
			for j in range(self.degree-1):
				vx[i, j] = self.x[i] ** (self.degree - 1 - j)
		self.w = np.dot(np.dot(np.linalg.inv(np.dot(vx.T, vx) + vreg), vx.T), self.y)

	def plot(self):
		poly_fun = np.poly1d(self.w)
		x = np.linspace(0, 1, 1000)
		plt.ylim(-1.5, 1.5)
		plt.title('N=%s, M=%s, lambda= %s' % (self.n, self.degree-1, self.ex))
		plt.plot(x, np.sin(x * 2 * np.pi), 'g', label = 'y = sin(x)')
		plt.plot(x, poly_fun(x), 'r', label = 'curve fitting')
		plt.plot(self.x, self.y, 'o')
		plt.legend()
		plt.show()


if __name__ == '__main__':

	cf1 = CurveFitting(10, 3, 0, 0)
	cf2 = CurveFitting(10, 9, 0, 0)
	cf3 = CurveFitting(15, 9, 0, 0)
	cf4 = CurveFitting(100, 9, 0, 0)
	cf5 = CurveFitting(10, 9, 1, 0)

	cf1.generate_sin()
	cf2.generate_sin()
	cf3.generate_sin()
	cf4.generate_sin()
	cf5.generate_sin()

	cf1.poly_fit()
	cf2.poly_fit()
	cf3.poly_fit()
	cf4.poly_fit()
	cf5.poly_fit()

	cf1.plot()
	cf2.plot()
	cf3.plot()
	cf4.plot()
	cf5.plot()




