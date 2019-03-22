'''
Validation accuracy
'''
from kernels import *
from classifiers import *
import time
import pandas as pd
import numpy as np 



def make_dataset_uniform_experiment(save_dir="../datasets"):

	Xtr0 = pd.read_csv(os.path.join(save_dir, "Xtr0.csv"))
	Xtr1 = pd.read_csv(os.path.join(save_dir, "Xtr1.csv"))
	Xtr2 = pd.read_csv(os.path.join(save_dir, "Xtr2.csv"))

	Ytr0 = pd.read_csv(os.path.join(save_dir, "Ytr0.csv"))
	Ytr1 = pd.read_csv(os.path.join(save_dir, "Ytr1.csv"))
	Ytr2 = pd.read_csv(os.path.join(save_dir, "Ytr2.csv"))

	Xtr = pd.concat([Xtr0, Xtr1, Xtr2])
	Xtr.reset_index(drop=True, inplace=True)

	Ytr = pd.concat([Ytr0, Ytr1, Ytr2])
	Ytr.reset_index(drop=True, inplace=True)

	Xtrain = Xtr.loc[:5000 - 1, :]
	Xval = Xtr.loc[5000:, :]
	Ytrain = Ytr.loc[:5000 - 1, :]
	Yval = Ytr.loc[5000:, :]

	Xtrain.reset_index(drop=True, inplace=True)
	Xval.reset_index(drop=True, inplace=True)
	Ytrain.reset_index(drop=True, inplace=True)
	Yval.reset_index(drop=True, inplace=True)

	Ytrain.loc[Ytrain.Bound == 0, "Bound"] = - 1
	Yval.loc[Yval.Bound == 0, "Bound"] = - 1

	return Xtrain, Ytrain, Xval, Yval

def make_dataset_specific_experiment(save_dir="../datasets"):

	Xtr0 = pd.read_csv(os.path.join(save_dir, "Xtr0.csv"))
	Xtr1 = pd.read_csv(os.path.join(save_dir, "Xtr1.csv"))
	Xtr2 = pd.read_csv(os.path.join(save_dir, "Xtr2.csv"))

	Ytr0 = pd.read_csv(os.path.join(save_dir, "Ytr0.csv"))
	Ytr1 = pd.read_csv(os.path.join(save_dir, "Ytr1.csv"))
	Ytr2 = pd.read_csv(os.path.join(save_dir, "Ytr2.csv"))


	Xtrain0 = Xtr0.loc[:1660 - 1, :]
	Ytrain0 = Ytr0.loc[:1660 - 1, :]
	Xval0 = Xtr0.loc[1660:2000 - 1, :]
	Yval0 = Ytr0.loc[1660:2000 - 1, :]

	Xtrain1 = Xtr1.loc[:1660 - 1, :]
	Ytrain1 = Ytr1.loc[:1660 - 1, :]
	Xval1 = Xtr1.loc[1660:2000 - 1, :]
	Yval1 = Ytr1.loc[1660:2000 - 1, :]

	Xtrain2 = Xtr2.loc[:1660 - 1, :]
	Ytrain2 = Ytr2.loc[:1660 - 1, :]
	Xval2 = Xtr2.loc[1660:2000 - 1, :]
	Yval2 = Ytr2.loc[1660:2000 - 1, :]

	
	Xtrain0.reset_index(drop=True, inplace=True)
	Ytrain0.reset_index(drop=True, inplace=True)
	Xval0.reset_index(drop=True, inplace=True)
	Yval0.reset_index(drop=True, inplace=True)

	Xtrain1.reset_index(drop=True, inplace=True)
	Ytrain1.reset_index(drop=True, inplace=True)
	Xval1.reset_index(drop=True, inplace=True)
	Yval1.reset_index(drop=True, inplace=True)

	Xtrain2.reset_index(drop=True, inplace=True)
	Ytrain2.reset_index(drop=True, inplace=True)
	Xval2.reset_index(drop=True, inplace=True)
	Yval2.reset_index(drop=True, inplace=True)


	Ytrain0.loc[Ytrain0.Bound == 0, "Bound"] = - 1
	Yval0.loc[Yval0.Bound == 0, "Bound"] = - 1
	Ytrain1.loc[Ytrain1.Bound == 0, "Bound"] = - 1
	Yval1.loc[Yval1.Bound == 0, "Bound"] = - 1
	Ytrain2.loc[Ytrain2.Bound == 0, "Bound"] = - 1
	Yval2.loc[Yval2.Bound == 0, "Bound"] = - 1

	Xtrain = [Xtrain0, Xtrain1, Xtrain2]
	Ytrain = [Ytrain0, Ytrain1, Ytrain2]
	Xval = [Xval0, Xval1, Xval2]
	Yval = [Yval0, Yval1, Yval2]

	return Xtrain, Ytrain, Xval, Yval

def experiment_uniform(Xtrain, Ytrain, Xval, Yval, 
					   params, matrix_is_ready=False,
					   kernel_path="../kernels_with_matrix/.pkl"):
	Kernel = params["kernel"]
	if Kernel == SpectrumKernel:
		k = params["k"]

	if LAKernel in Kernel:
		beta = params["beta"]

	Clf = params["clf"]
	if Clf == KernelLogisticRegression:
		lambda_ = params["lambda_"]
		save_kernel_with_matrix = params["save_kernel_with_matrix"]
		verbose = params["verbose"]
		tolerance = params["tolerance"]
	elif Clf == KernelSVM:
		C = params["C"]
		save_kernel_with_matrix = params["save_kernel_with_matrix"]
		verbose = params["verbose"]

	if Kernel == SpectrumKernel:
		kernel = Kernel(k=k)
	elif Kernel == LAKernel:
		kernel = Kernel(beta=beta)

	if Clf == KernelLogisticRegression:
		clf = Clf(kernel, lambda_=lambda_, 
				  save_kernel_with_matrix=save_kernel_with_matrix,
				  verbose=verbose,
				  dataset_idx=-1,
				  tolerance=tolerance)
	elif Clf == KernelSVM:
		clf = Clf(kernel, C=C,
				  save_kernel_with_matrix=save_kernel_with_matrix,
				  verbose=verbose,
				  dataset_idx=-1)

	if not matrix_is_ready:
		clf.fit(Xtrain, Ytrain.Bound)
	else:
		kernel = Kernel.load_kernel(kernel_path)
		clf.fit_matrix(kernel, Ytrain.Bound)

	print("classifier already fit")

	print("The score is:")
	print(clf.score(Xval, Yval.Bound))

def experiment_specific(Xtrain, Ytrain, Xval, Yval,
						params, matrix_is_ready=False,
						kernel_path="../kernels_with_matrix/.pkl",
						ensemble=False):
	Kernel = params["kernel"]
	if SpectrumKernel in Kernel:
		k = params["k"]

	if LAKernel in Kernel:
		beta = params["beta"]

	Clf = params["clf"]
	if KernelLogisticRegression in Clf:
		lambda_ = params["lambda_"]
		save_kernel_with_matrix = params["save_kernel_with_matrix"]
		verbose = params["verbose"]
		tolerance = params["tolerance"]

	if KernelSVM in Clf:
		C = params["C"]
		save_kernel_with_matrix = params["save_kernel_with_matrix"]
		verbose = params["verbose"]

	score = np.zeros(3)
	if ensemble:
		predictions = []
	for i in range(3):
		print("i =", i)
		if Kernel[i] == SpectrumKernel:
			kernel = Kernel[i](k=k[i])
		elif Kernel[i] == LAKernel:
			kernel = Kernel[i](beta=beta[i])

		if Clf[i] == KernelLogisticRegression:
			clf = Clf[i](kernel, lambda_=lambda_[i], 
						 save_kernel_with_matrix=save_kernel_with_matrix[i],
						 verbose=verbose[i],
						 dataset_idx=i,
						 tolerance=tolerance[i])
		elif Clf[i] == KernelSVM:
			clf = Clf[i](kernel, C=C[i],
						 save_kernel_with_matrix=save_kernel_with_matrix[i],
						 verbose=verbose[i],
						 dataset_idx=i)

		if not matrix_is_ready:
			clf.fit(Xtrain[i], Ytrain[i].Bound)
		else:
			path = kernel_path[:- 4] + "_" + str(i) + ".pkl"
			kernel = Kernel[i].load_kernel(path)
			clf.fit_matrix(kernel, Ytrain[i].Bound)

		if not ensemble:
			score[i] = clf.score(Xval[i], Yval[i].Bound)
		else:
			predictions.append(clf.predict(Xval[i]))
	if not ensemble:
		print("The score is:")
		print(score.mean())
	else:
		prediction = np.concatenate((predictions[0], 
									 predictions[1], 
									 predictions[2]), axis=0)
		return prediction

def experiment_specific_single(Xtrain, Ytrain, Xval, Yval, i,
							   params, matrix_is_ready=False,
							   kernel_path="../kernels_with_matrix/.pkl"):
	Kernel = params["kernel"]
	if Kernel[i] == SpectrumKernel:
		k = params["k"]

	if LAKernel in Kernel:
		beta = params["beta"]

	Clf = params["clf"]
	if Clf[i] == KernelLogisticRegression:
		lambda_ = params["lambda_"]
		save_kernel_with_matrix = params["save_kernel_with_matrix"]
		verbose = params["verbose"]
		tolerance = params["tolerance"]
	elif Clf[i] == KernelSVM:
		C = params["C"]
		save_kernel_with_matrix = params["save_kernel_with_matrix"]
		verbose = params["verbose"]

	if Kernel[i] == SpectrumKernel:
		kernel = Kernel[i](k=k[i])
	elif Kernel[i] == LAKernel:
			kernel = Kernel[i](beta=beta[i])

	if Clf[i] == KernelLogisticRegression:
		clf = Clf[i](kernel, lambda_=lambda_[i] , 
					 save_kernel_with_matrix=save_kernel_with_matrix[i],
					 verbose=verbose[i],
					 dataset_idx=i,
					 tolerance=tolerance[i])
	elif Clf[i] == KernelSVM:
		clf = Clf[i](kernel, C=C[i],
					 save_kernel_with_matrix=save_kernel_with_matrix[i],
					 verbose=verbose[i],
					 dataset_idx=i)

	if not matrix_is_ready:
		clf.fit(Xtrain[i], Ytrain[i].Bound)
	else:
		path = kernel_path[:- 4] + "_" + str(i) + ".pkl"
		kernel = Kernel[i].load_kernel(path)
		clf.fit_matrix(kernel, Ytrain[i].Bound)


	print("The score is:")
	print(clf.score(Xval[i], Yval[i].Bound))


np.random.seed(0)
pd.options.mode.chained_assignment = None


t0 = time.time()

use_ensemble = True

if not use_ensemble:

	params = {"kernel" : [SpectrumKernel] * 3, 
			  "clf" : [KernelLogisticRegression] * 3, 
			  "k" : [9, 9, 9], "lambda_" : [1e-4, 1e-4, 1e-4], "C" : [None, None, None],
			  "beta": [0.5] * 3,
			  "save_kernel_with_matrix" : [True] * 3, "verbose" : [False] * 3, 
			  "tolerance" : [1e-5] * 3}

	use_specific = True

	if use_specific:
		Xtrain, Ytrain, Xval, Yval = make_dataset_specific_experiment(save_dir="../datasets")

		experiment_specific(Xtrain, Ytrain, Xval, Yval, 
							params, matrix_is_ready=False,
							kernel_path="../kernels_with_matrix/.pkl")
	else:
		Xtrain, Ytrain, Xval, Yval = make_dataset_uniform_experiment(save_dir="../datasets")

		experiment_uniform(Xtrain, Ytrain, Xval, Yval, 
						   params, matrix_is_ready=False,
						   kernel_path="../kernels_with_matrix/.pkl")
else:
	Xtrain, Ytrain, Xval, Yval = make_dataset_specific_experiment(save_dir="../datasets")
	chunk_size = Xval[0].shape[0] 

	ensemble_result = np.zeros(3 * chunk_size)

	# dataset 0
	dataset_idx = 0
	print(dataset_idx)

	kernel = SpectrumKernel(k=9)
	clf = KernelLogisticRegression(kernel, lambda_=1e-4, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx,
				  tolerance=1e-5)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[:chunk_size] += clf.predict(Xval[dataset_idx])

	kernel = SpectrumKernel(k=12)
	clf = KernelLogisticRegression(kernel, lambda_=1e-2, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx,
				  tolerance=1e-5)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[:chunk_size] += clf.predict(Xval[dataset_idx])

	kernel = SpectrumKernel(k=12)
	clf = KernelSVM(kernel, C=1, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[:chunk_size] += clf.predict(Xval[dataset_idx])

	# dataset 1
	dataset_idx = 1
	print(dataset_idx)

	kernel = SpectrumKernel(k=9)
	clf = KernelLogisticRegression(kernel, lambda_=1e-4, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx,
				  tolerance=1e-5)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[chunk_size:chunk_size*2] += clf.predict(Xval[dataset_idx])
	
	kernel = SpectrumKernel(k=9)
	clf = KernelLogisticRegression(kernel, lambda_=1e-3, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx,
				  tolerance=1e-5)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[chunk_size:chunk_size*2] += clf.predict(Xval[dataset_idx])

	kernel = SpectrumKernel(k=8)
	clf = KernelSVM(kernel, C=1e-2, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[chunk_size:chunk_size*2] += clf.predict(Xval[dataset_idx])
	
	# dataset 2
	dataset_idx = 2
	print(dataset_idx)

	kernel = SpectrumKernel(k=9)
	clf = KernelLogisticRegression(kernel, lambda_=1e-4, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx,
				  tolerance=1e-5)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[chunk_size*2:chunk_size*3] += clf.predict(Xval[dataset_idx])
	
	kernel = SpectrumKernel(k=8)
	clf = KernelLogisticRegression(kernel, lambda_=1e-4, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx,
				  tolerance=1e-5)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[chunk_size*2:chunk_size*3] += clf.predict(Xval[dataset_idx])

	kernel = SpectrumKernel(k=12)
	clf = KernelSVM(kernel, C=1e-2, 
				  save_kernel_with_matrix=True,
				  verbose=False,
				  dataset_idx=dataset_idx)
	clf.fit(Xtrain[dataset_idx], Ytrain[dataset_idx].Bound)
	ensemble_result[chunk_size*2:chunk_size*3] += clf.predict(Xval[dataset_idx])
	
	# ensemble
	ensemble_result = ensemble_voting(ensemble_result)
	scores = np.zeros(3)
	for i in range(3):
		scores[i] = (ensemble_result[i * chunk_size : (i + 1) * \
					chunk_size] == Yval[i].Bound).sum() / chunk_size
	print("The score is:")
	print(scores.mean())


	
print("Done in %.4f s." % (time.time() - t0))






