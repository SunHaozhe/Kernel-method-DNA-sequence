from algorithms import *
from kernels import *
import time
import pandas as pd

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


kernel = SpectrumKernel(3)
clf = KernelLogisticRegression(kernel, lambda_=0.01)
clf.fit(Xtrain, Ytrain, tolerance=1e-5)

print(clf.score(Xval, Yval))






print("Done in %.4f s." % (time.time() - t0))