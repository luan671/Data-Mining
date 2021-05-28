import numpy as np #import numpy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

cleveland = np.genfromtxt("D:\Data Mining\HW3\cleveland.txt",delimiter=None)
X = cleveland[:,0:14] # load data and split into features, targets
Y = cleveland[:,14]
print(X.shape)

X, Y = shuffle(X, Y)