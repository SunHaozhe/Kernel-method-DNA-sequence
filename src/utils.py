import numpy as np 
import pandas as pd

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
	'''
	Numerically stable sigmoid function, prevents overflow in exp
	'''
	positive = x >= 0
	negative = x < 0
	xx = 1 / (1 + np.exp(- x[positive]))
	x[positive] = 1 / (1 + np.exp(- x[positive]))
	z = np.exp(x[negative])
	x[negative] = z / (z + 1)
	return x

def submit(x, verbose=True):
	'''
	substitute -1 by 0, then save to a csv file in the appropriate format
	'''
	x = pd.DataFrame(x)
	x.reset_index(inplace=True)
	x.columns = ["Id", "Bound"]
	x.loc[x.Bound == - 1, "Bound"] = 0
	x.to_csv("../Yte.csv", index=None)
	if verbose:
		print("Yte saved.")