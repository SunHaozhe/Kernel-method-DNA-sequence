from classifiers import *
from kernels import *
import time, os
import pandas as pd
import numpy as np 

np.random.seed(0)
pd.options.mode.chained_assignment = None


t0 = time.time()

save_dir = "../datasets"

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


kernel = SpectrumKernel(k=11)
clf = KernelLogisticRegression(kernel, lambda_=1e-6, 
							   save_kernel_matrix=True,
							   verbose=True,
							   tolerance=1e-5)


matrix_is_ready = False

if not matrix_is_ready:
	clf.fit(Xtr, Ytr.Bound)
else:
	kernel = Kernel.load_kernel("../kernels_with_matrix/.pkl")
	clf.fit_matrix(kernel, Ytrain.Bound)

print("classifier already fit")

prediction = clf.predict(Xte)

submit(prediction)

print("Done in %.4f s." % (time.time() - t0))






