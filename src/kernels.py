import numpy as np


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

	def K(self, item1, item2):
		raise NotImplementedError("K not implemented")

class SpectrumKernel(Kernel):
	def __init__(self, k):
		super().__init__()
		self.k = k

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







