from algorithms import *
from kernels import *
import time
import pandas as pd

pd.options.mode.chained_assignment = None


t0 = time.time()

Xtr0 = pd.read_csv("../datasets/Xtr0.csv")
Xtr1 = pd.read_csv("../datasets/Xtr1.csv")
Xtr2 = pd.read_csv("../datasets/Xtr2.csv")

Ytr0 = pd.read_csv("../datasets/Ytr0.csv")
Ytr1 = pd.read_csv("../datasets/Ytr1.csv")
Ytr2 = pd.read_csv("../datasets/Ytr2.csv")

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



kernel = SpectrumKernel(k=3)
clf = KernelLogisticRegression(kernel, lambda_=0.01, 
							   save_kernel_matrix=True,
							   verbose=True,
							   tolerance=1e-5)

#clf.fit(Xtrain, Ytrain.Bound)
kernel_matrix = kernel.load_kernel_matrix("../kernel_matrices/spectrum_kernel_k_3_5000.pkl")
clf.fit_matrix(kernel_matrix, Xtrain, Ytrain.Bound)
print("classifier already fit")

print("")
print("The score is:")
print(clf.score(Xval, Yval.Bound))


print("Done in %.4f s." % (time.time() - t0))