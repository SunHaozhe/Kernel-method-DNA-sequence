import numpy as np 
from kernels import *
from utils import *

class KernelClassifier:
	'''
	Abstract class representing general classifier using kernel methods
	'''
	def __init__(self, kernel):
		self.kernel = kernel

	def predict_one_instance(self, x_test):
		vector = self.kernel.test(self.X_train, x_test)
		return self.alpha.dot(vector)

	def predict(self, X_test):
		prediction = np.zeros(X_test.shape[0])
		for i in range(X_test.shape[0]):
			prediction[i] = self.predict_one_instance(X_test.loc[i, X_test.columns[1]])
		return prediction

	def score(self, X, y):
		prediction = self.predict(X)
		return (prediction == y).sum() / prediction.size

	def fit(self, X, y):
		raise NotImplementedError("fit not implemented")


class KernelLogisticRegression(KernelClassifier):
	'''
	Kernel Logistic Regression
	'''
	def __init__(self, kernel, lambda_):
		super().__init__(kernel)
		self.lambda_ = lambda_

	def fit(self, X, y, tolerance=1e-5):
		self.X_train = X
		self.K = self.kernel.build_matrix(X)
		self.alpha = np.random.rand(self.K.shape[0])
		previous = self.alpha
		while True:
			m = K.dot(self.alpha)
			P = - sigmoid(- y * m)
			W = sigmoid(m) * sigmoid(- m)
			z = m - P * y / W 
			W = np.diag(W)
			self.alpha = solveWKRR(self.K, W, z, self.lambda_)

			if np.linalg.norm(previous, self.alpha) <= tolerance:
				break

