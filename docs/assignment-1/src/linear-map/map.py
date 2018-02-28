# ========================================
# [] File Name : map.py
#
# [] Creation Date : February 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Implementation of a linear transformation on a multivariate Gaussian distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

MEAN_VECTOR = np.array([
    0,0
])
COV_MATRIX = np.array([
    [1, 0],
    [0, 1]
])

print("Mean Vector Shape: ", MEAN_VECTOR.shape)
print("Cov Matrix Shape: ", COV_MATRIX.shape)

GAUSSIAN_SAMPLES = np.random.multivariate_normal(MEAN_VECTOR, COV_MATRIX, size=500)
X_SEQUENCE = np.random.normal(size=500)

FIG = plt.figure()
AX = Axes3D(FIG)

AX.scatter(X_SEQUENCE, GAUSSIAN_SAMPLES[:, 0], GAUSSIAN_SAMPLES[:, 1])
plt.show()

#print(GAUSSIAN_SAMPLES)