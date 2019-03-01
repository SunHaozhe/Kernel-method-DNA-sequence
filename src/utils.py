import numpy as np 

def solveWKRR(K, W, Y, lambda_):
	'''
	solves Weighted Kernel Ridge Regression
	K: Kernel matrix
	W: diagonal weight matrix
	Y: label vector
	'''
	n = W.shape[0]
	Wsqrt = np.diag(np.sqrt(np.diag(W)))
	return Wsqrt.dot(np.linalg.solve(Wsqrt.dot(K).dot(Wsqrt) \
		   + n * lambda_ * np.identity(n), Wsqrt.dot(Y)))

def sigmoid(x):
	return 1 / (1 + np.exp(- x))