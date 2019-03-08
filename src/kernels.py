import numpy as np
import pickle


class Kernel:
	def __init__(self):
		pass

	def build_matrix(self, X):
		n = X.shape[0]
		output = np.zeros((n, n))
		for i in range(n):
			for j in range(i, n):
				value = self.K(X.loc[i, X.columns[1]], X.loc[j, X.columns[1]])
				if i != j:
					output[i, j] = output[j, i] = value
				else:
					output[i, j] = value
		return output

	def test(self, X_train, x_test):
		n = X_train.shape[0]
		output = np.zeros(n)
		for i in range(n):
			output[i] = self.K(X_train.loc[i, X_train.columns[1]], x_test)
		return output

	@staticmethod
	def save_kernel_matrix(save_path, kernel_matrix):
		with open(save_path, "wb") as f:
			pickle.dump(kernel_matrix, f)

	@staticmethod
	def load_kernel_matrix(save_path):
		with open(save_path, "rb") as f:
			kernel_matrix = pickle.load(f)
		return kernel_matrix

	def K(self, item1, item2):
		raise NotImplementedError("K not implemented")


class SpectrumKernel(Kernel):
	def __init__(self, k):
		super().__init__()
		self.k = k
		self.name = "spectrum_kernel_k_" + str(k)

	def K(self, item1, item2):
		assert len(item1) >= self.k and len(item2) >= self.k, "sequences too short"
		dict1 = self.make_dict(item1)
		dict2 = self.make_dict(item2)
		keys = list(set(dict1.keys()) & set(dict2.keys()))
		output = 0
		for key in keys:
			output += dict1[key] * dict2[key]
		return output
		

	def make_dict(self, item):
		dict_ = {}
		for i in range(len(item) - self.k + 1):
			target = item[i : i + self.k]
			if target not in dict_.keys():
				dict_[target] = 1
			else:
				dict_[target] += 1
		return dict_


class SpectrumKernelQuick(Kernel):
	'''
	Optimized version of spectrum kernel
	'''
	def __init__(self, k):
		super().__init__()
		self.k = k
		self.name = "spectrum_kernel_k_" + str(k)

	def build_matrix(self, X):
		n = X.shape[0]
		output = np.zeros((n, n))
		self.train_dicts = []
		for i in range(n):
			item = X.loc[i, X.columns[1]]
			assert len(item) >= self.k, "sequences too short"
			self.train_dicts.append(self.make_dict(item))

		for i in range(n):
			for j in range(i, n):
				value = self.inner_product_dict(self.train_dicts[i], self.train_dicts[j]) 
				if i != j:
					output[i, j] = output[j, i] = value
				else:
					output[i, j] = value
		return output

	def make_test_dicts(self, X_test):
		n = X_test.shape[0]
		self.test_dicts = []
		for i in range(n):
			item = X_test.loc[i, X_test.columns[1]]
			assert len(item) >= self.k, "sequences too short"
			self.test_dicts.append(self.make_dict(item))

	def test(self, idx):
		'''
		idx: index of test example in the test set
		'''
		n = len(self.train_dicts)
		output = np.zeros(n)
		for i in range(n):
			output[i] = self.inner_product_dict(self.train_dicts[i], self.test_dicts[idx])
		return output

	def inner_product_dict(self, dict1, dict2):
		keys = list(set(dict1.keys()) & set(dict2.keys()))
		output = 0
		for key in keys:
			output += dict1[key] * dict2[key]
		return output
		

	def make_dict(self, item):
		dict_ = {}
		for i in range(len(item) - self.k + 1):
			target = item[i : i + self.k]
			if target not in dict_.keys():
				dict_[target] = 1
			else:
				dict_[target] += 1
		return dict_






