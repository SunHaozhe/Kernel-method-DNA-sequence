import numpy as np 
from kernels import *
from utils import *
import os

class KernelClassifier:
	'''
	Abstract class representing general classifier using kernel methods
	'''
	def __init__(self, kernel, 
		         save_kernel_matrix=False, 
		         verbose=False,
		         tolerance=1e-5):
		self.kernel = kernel
		self.save_kernel_matrix = save_kernel_matrix
		self.verbose = verbose
		self.tolerance = tolerance
		self.kernel_matrices_dir = "../kernel_matrices"

		if not os.path.exists(self.kernel_matrices_dir):
			os.makedirs(self.kernel_matrices_dir)

	def predict_one_instance(self, x_test):
		vector = self.kernel.test(self.X_train, x_test)
		return self.alpha.dot(vector)

	def predict(self, X_test):
		prediction = np.zeros(X_test.shape[0])
		for i in range(X_test.shape[0]):
			prediction[i] = self.predict_one_instance(X_test.loc[i, X_test.columns[1]])
		output = np.zeros_like(prediction, dtype=int)
		output[prediction < 0] = - 1
		output[prediction >= 0] = 1
		return output

	def score(self, X, y):
		prediction = self.predict(X)
		return (prediction == y).sum() / prediction.size

	def fit(self, X, y):
		self.K = self.kernel.build_matrix(X)

		if self.verbose:
			print("kernel matrix built")
			print(self.K)

		if self.save_kernel_matrix:
			save_path = os.path.join(self.kernel_matrices_dir, 
									 self.kernel.name + "_"+ \
									 str(self.K.shape[0]) + ".pkl")
			self.kernel.save_kernel_matrix(save_path, self.K)

		self.fit_matrix(self.K, X, y)

	def fit_matrix(self, K, X, y):
		self.K = K
		self.X_train = X
		self.fit_matrix_(K, y)

	def fit_matrix_(self, K, y):
		raise NotImplementedError("__fit_matrix not implemented")


class KernelLogisticRegression(KernelClassifier):
	'''
	Kernel Logistic Regression
	'''
	def __init__(self, kernel, lambda_, 
		         save_kernel_matrix=False, 
		         verbose=False,
		         tolerance=1e-5):
		super().__init__(kernel, save_kernel_matrix, verbose, tolerance)
		self.lambda_ = lambda_

	def fit_matrix_(self, K, y):
		self.alpha = np.random.rand(self.K.shape[0])
		previous = self.alpha
		while True:
			m = self.K.dot(self.alpha)
			P = - sigmoid(- y * m)
			W = sigmoid(m) * sigmoid(- m)
			z = m - P * y / W 
			W = np.diag(W)
			self.alpha = solveWKRR(self.K, W, z, self.lambda_)
			
			error = np.linalg.norm(previous - self.alpha)
			if self.verbose:
				print("error =", error)
			if error <= self.tolerance:
				break
			previous = self.alpha


class KernelLogisticRegressionQuick(KernelClassifier):
	'''
	Kernel Logistic Regression (optimized version)
	'''
	def __init__(self, kernel, lambda_, 
		         save_kernel_matrix=False, 
		         verbose=False,
		         tolerance=1e-5):
		super().__init__(kernel, save_kernel_matrix, verbose, tolerance)
		self.lambda_ = lambda_

	def predict_one_instance(self, idx):
		vector = self.kernel.test(idx)
		return self.alpha.dot(vector)

	def predict(self, X_test):
		self.kernel.make_test_dicts(X_test)
		prediction = np.zeros(X_test.shape[0])
		for idx in range(X_test.shape[0]):
			prediction[idx] = self.predict_one_instance(idx)
		output = np.zeros_like(prediction, dtype=int)
		output[prediction < 0] = - 1
		output[prediction >= 0] = 1
		return output

	def fit_matrix(self, K, X, y):
		self.K = K
		self.fit_matrix_(K, y)

	def fit_matrix_(self, K, y):
		self.alpha = np.random.rand(self.K.shape[0])
		previous = self.alpha
		while True:
			m = self.K.dot(self.alpha)
			P = - sigmoid(- y * m)
			W = sigmoid(m) * sigmoid(- m)
			z = m - P * y / W 
			W = np.diag(W)
			self.alpha = solveWKRR(self.K, W, z, self.lambda_)
			
			error = np.linalg.norm(previous - self.alpha)
			if self.verbose:
				print("error =", error)
			if error <= self.tolerance:
				break
			previous = self.alpha

