import numpy as np
import pickle

class Kernel():
	def __init__(self):
		pass

	def build_matrix(self, X):
		raise NotImplementedError("build_matrix not implemented.")

	def test(self, x_test):
		raise NotImplementedError("test not implemented.")

	def save_kernel(self, save_path):
		with open(save_path, "wb") as f:
			pickle.dump(self, f)

	@staticmethod
	def load_kernel(save_path):
		with open(save_path, "rb") as f:
			kernel_matrix = pickle.load(f)
		return kernel_matrix


class KernelImplicit(Kernel):
	'''
	kernels with implicit inner products
	'''
	def __init__(self):
		super().__init__()

	def build_matrix(self, X):
		self.X_train = X
		n = X.shape[0]
		output = np.zeros((n, n))
		for i in range(n):
			for j in range(i, n):
				value = self.K(X.loc[i, X.columns[1]], X.loc[j, X.columns[1]])
				output[i, j] = output[j, i] = value
		self.K_matrix = output

	def test(self, x_test):
		n = self.X_train.shape[0]
		output = np.zeros(n)
		for i in range(n):
			output[i] = self.K(self.X_train.loc[i, self.X_train.columns[1]], x_test)
		return output


	def K(self, item1, item2):
		raise NotImplementedError("K not implemented")


class KernelExplicit(Kernel):
	'''
	kernels with explicit inner products
	'''
	def __init__(self):
		super().__init__()

	def build_matrix(self, X):
		n = X.shape[0]
		output = np.zeros((n, n))
		self.train_phi = []
		for i in range(n):
			item = X.loc[i, X.columns[1]]
			self.train_phi.append(self.make_phi(item)) #self.make_dict(item)

		for i in range(n):
			for j in range(i, n):
				value = self.inner_product_phi(self.train_phi[i], self.train_phi[j])
				output[i, j] = output[j, i] = value
		self.K_matrix = output

	def test(self, x_test):
		'''
		x_test: index of test example in the test set
		'''
		n = len(self.train_phi)
		output = np.zeros(n)
		for i in range(n):
			output[i] = self.inner_product_phi(self.train_phi[i], self.test_phi[x_test])
		return output

	def make_test_phi(self, X_test):
		n = X_test.shape[0]
		self.test_phi = []
		for i in range(n):
			item = X_test.loc[i, X_test.columns[1]]
			self.test_phi.append(self.make_phi(item))

	def make_phi(self, item):
		raise NotImplementedError("make_phi not implemented.")

	def inner_product_phi(self, phi1, phi2):
		raise NotImplementedError("inner_product_phi not implemented.")


class SpectrumKernelSlow(KernelImplicit):
	'''
	This is depreciated because spectrum kernel corresponds to explicit inner products
	'''
	def __init__(self, k):
		super().__init__()
		self.k = k
		self.name = "spectrum_kernel_k_" + str(k)

	def K(self, item1, item2):
		assert len(item1) >= self.k and len(item2) >= self.k, "sequences too short"
		dict1 = self.make_phi(item1)
		dict2 = self.make_phi(item2)
		keys = list(set(dict1.keys()) & set(dict2.keys()))
		output = 0
		for key in keys:
			output += dict1[key] * dict2[key]
		return output
		

	def make_phi(self, item):
		dict_ = {}
		for i in range(len(item) - self.k + 1):
			target = item[i : i + self.k]
			if target not in dict_.keys():
				dict_[target] = 1
			else:
				dict_[target] += 1
		return dict_


class SpectrumKernel(KernelExplicit):
	'''
	spectrum kernel corresponds to explicit inner products
	'''
	def __init__(self, k):
		super().__init__()
		self.k = k
		self.name = "spectrum_kernel_k_" + str(k)

	def make_phi(self, item):
		assert len(item) >= self.k, "sequences too short"
		dict_ = {}
		for i in range(len(item) - self.k + 1):
			target = item[i : i + self.k]
			if target not in dict_.keys():
				dict_[target] = 1
			else:
				dict_[target] += 1
		return dict_

	def inner_product_phi(self, dict1, dict2):
		keys = list(set(dict1.keys()) & set(dict2.keys()))
		output = 0
		for key in keys:
			output += dict1[key] * dict2[key]
		return output
		






