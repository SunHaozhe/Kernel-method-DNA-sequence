from kernels import *
from classifiers import *
import time, os
import pandas as pd
import numpy as np 

def make_dataset_uniform_start(save_dir="../datasets"):

	Xtr0 = pd.read_csv(os.path.join(save_dir, "Xtr0.csv"))
	Xtr1 = pd.read_csv(os.path.join(save_dir, "Xtr1.csv"))
	Xtr2 = pd.read_csv(os.path.join(save_dir, "Xtr2.csv"))

	Ytr0 = pd.read_csv(os.path.join(save_dir, "Ytr0.csv"))
	Ytr1 = pd.read_csv(os.path.join(save_dir, "Ytr1.csv"))
	Ytr2 = pd.read_csv(os.path.join(save_dir, "Ytr2.csv"))

	Xte0 = pd.read_csv(os.path.join(save_dir, "Xte0.csv"))
	Xte1 = pd.read_csv(os.path.join(save_dir, "Xte1.csv"))
	Xte2 = pd.read_csv(os.path.join(save_dir, "Xte2.csv"))


	Xtr = pd.concat([Xtr0, Xtr1, Xtr2])
	Xtr.reset_index(drop=True, inplace=True)

	Ytr = pd.concat([Ytr0, Ytr1, Ytr2])
	Ytr.reset_index(drop=True, inplace=True)

	Xte = pd.concat([Xte0, Xte1, Xte2])
	Xte.reset_index(drop=True, inplace=True)

	Ytr.loc[Ytr.Bound == 0, "Bound"] = - 1

	return Xtr, Ytr, Xte

def make_dataset_specific_start(save_dir="../datasets"):
	Xtr0 = pd.read_csv(os.path.join(save_dir, "Xtr0.csv"))
	Xtr1 = pd.read_csv(os.path.join(save_dir, "Xtr1.csv"))
	Xtr2 = pd.read_csv(os.path.join(save_dir, "Xtr2.csv"))

	Ytr0 = pd.read_csv(os.path.join(save_dir, "Ytr0.csv"))
	Ytr1 = pd.read_csv(os.path.join(save_dir, "Ytr1.csv"))
	Ytr2 = pd.read_csv(os.path.join(save_dir, "Ytr2.csv"))

	Xte0 = pd.read_csv(os.path.join(save_dir, "Xte0.csv"))
	Xte1 = pd.read_csv(os.path.join(save_dir, "Xte1.csv"))
	Xte2 = pd.read_csv(os.path.join(save_dir, "Xte2.csv"))

	Ytr0.loc[Ytr0.Bound == 0, "Bound"] = - 1
	Ytr1.loc[Ytr1.Bound == 0, "Bound"] = - 1
	Ytr2.loc[Ytr2.Bound == 0, "Bound"] = - 1

	Xtr = [Xtr0, Xtr1, Xtr2]
	Ytr = [Ytr0, Ytr1, Ytr2]
	Xte = [Xte0, Xte1, Xte2]

	return Xtr, Ytr, Xte

def start_uniform(Xtr, Ytr, Xte, 
				  params, matrix_is_ready=False,
				  kernel_path="../kernels_with_matrix/.pkl"):
	Kernel = params["kernel"]
	if Kernel == SpectrumKernel:
		k = params["k"]

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
		clf.fit(Xtr, Ytr.Bound)
	else:
		kernel = Kernel.load_kernel("../kernels_with_matrix/.pkl")
		clf.fit_matrix(kernel, Ytrain.Bound)

	print("classifier already fit")

	prediction = clf.predict(Xte)

	submit(prediction)

def start_specific(Xtr, Ytr, Xte, 
				   params, matrix_is_ready=False,
				   kernel_path="../kernels_with_matrix/.pkl"):
	Kernel = params["kernel"]
	if Kernel == SpectrumKernel:
		k = params["k"]

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

	predictions = []
	for i in range(3):
		print("i =", i)
		if Kernel == SpectrumKernel:
			kernel = Kernel(k=k[i])

		if Clf == KernelLogisticRegression:
			clf = Clf(kernel, lambda_=lambda_, 
					  save_kernel_with_matrix=save_kernel_with_matrix,
					  verbose=verbose,
					  dataset_idx=i,
					  tolerance=tolerance)
		elif Clf == KernelSVM:
			clf = Clf(kernel, C=C,
					  save_kernel_with_matrix=save_kernel_with_matrix,
					  verbose=verbose,
					  dataset_idx=i)

		if not matrix_is_ready:
			clf.fit(Xtr[i], Ytr[i].Bound)
		else:
			path = kernel_path[:- 4] + "_" + str(i) + ".pkl"
			kernel = Kernel.load_kernel(path)
			clf.fit_matrix(kernel, Ytr[i].Bound)

		predictions.append(clf.predict(Xte[i]))

	prediction = np.concatenate((predictions[0], 
								 predictions[1], 
								 predictions[2]), axis=0)
	submit(prediction)



np.random.seed(0)
pd.options.mode.chained_assignment = None


t0 = time.time()

params = {"kernel" : SpectrumKernel, "clf" : KernelLogisticRegression, "k" : [9] * 3,
		  "lambda_" : 1e-4, "save_kernel_with_matrix" : True, "verbose" : False,
		  "tolerance" : 1e-5}

use_specific = True

if use_specific:
	Xtr, Ytr, Xte = make_dataset_specific_start(save_dir="../datasets")

	start_specific(Xtr, Ytr, Xte, 
				   params, matrix_is_ready=False,
				   kernel_path="../kernels_with_matrix/.pkl")
else:
	Xtr, Ytr, Xte = make_dataset_uniform_start(save_dir="../datasets")

	start_uniform(Xtr, Ytr, Xte, 
				  params, matrix_is_ready=False,
				  kernel_path="../kernels_with_matrix/.pkl")



print("Done in %.4f s." % (time.time() - t0))






