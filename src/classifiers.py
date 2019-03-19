import numpy as np 
from kernels import *
from utils import *
import os

class KernelClassifier:
	'''
	Abstract class representing general classifier using kernel methods
	'''
	def __init__(self, kernel, 
		         save_kernel_with_matrix=False, 
		         verbose=False,
		         tolerance=1e-5):
		self.kernel = kernel
		self.save_kernel_with_matrix = save_kernel_with_matrix
		self.verbose = verbose
		self.tolerance = tolerance
		self.kernels_with_matrix_dir = "../kernels_with_matrix"

		if not os.path.exists(self.kernels_with_matrix_dir):
			os.makedirs(self.kernels_with_matrix_dir)

	def predict_one_instance(self, x_test):
		vector = self.kernel.test(x_test)
		return self.alpha.dot(vector)

	def predict(self, X_test):
		if isinstance(self.kernel, KernelImplicit):
			return self.predict_implicit(X_test)
		else:
			return self.predict_explicit(X_test)

	def predict_implicit(self, X_test):
		prediction = np.zeros(X_test.shape[0])
		for i in range(X_test.shape[0]):
			prediction[i] = self.predict_one_instance(X_test.loc[i, X_test.columns[1]])
		output = np.zeros_like(prediction, dtype=int)
		output[prediction < 0] = - 1
		output[prediction >= 0] = 1
		return output

	def predict_explicit(self, X_test):
		self.kernel.make_test_phi(X_test)
		prediction = np.zeros(X_test.shape[0])
		for idx in range(X_test.shape[0]):
			prediction[idx] = self.predict_one_instance(idx)
		output = np.zeros_like(prediction, dtype=int)
		output[prediction < 0] = - 1
		output[prediction >= 0] = 1
		return output

	def score(self, X, y):
		prediction = self.predict(X)
		return (prediction == y).sum() / prediction.size

	def fit(self, X, y):
		self.kernel.build_matrix(X)

		if self.verbose:
			print("kernel matrix built")
			print(self.kernel.K_matrix)

		if self.save_kernel_with_matrix:
			save_path = os.path.join(self.kernels_with_matrix_dir, 
									 self.kernel.name + "_"+ \
									 str(self.kernel.K_matrix.shape[0]) + ".pkl")
			self.kernel.save_kernel(save_path)

		self.fit_matrix(self.kernel, y)

	def fit_matrix(self, kernel, y):
		self.kernel = kernel
		self.fit_(y)

	def fit_matrix_(self, K, y):
		raise NotImplementedError("__fit_matrix not implemented")


class KernelLogisticRegression(KernelClassifier):
	'''
	Kernel Logistic Regression with kernel of explicit version
	'''
	def __init__(self, kernel, lambda_, 
		         save_kernel_with_matrix=False, 
		         verbose=False,
		         tolerance=1e-5):
		super().__init__(kernel, save_kernel_with_matrix, verbose, tolerance)
		self.lambda_ = lambda_

	def fit_(self, y):
		self.alpha = np.random.rand(self.kernel.K_matrix.shape[0])
		previous = self.alpha
		while True:
			m = self.kernel.K_matrix.dot(self.alpha)
			P = - sigmoid(- y * m)
			W = sigmoid(m) * sigmoid(- m)
			z = m - P * y / W 
			W = np.diag(W)
			self.alpha = solveWKRR(self.kernel.K_matrix, W, z, self.lambda_)
			
			error = np.linalg.norm(previous - self.alpha)
			if self.verbose:
				print("error =", error)
			if error <= self.tolerance:
				break
			previous = self.alpha










