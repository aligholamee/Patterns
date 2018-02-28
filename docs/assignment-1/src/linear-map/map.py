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
from matplotlib import cm
from scipy.stats import multivariate_normal

MEAN_VECTOR = np.array([0,0])

COV_MATRIX = np.array([
    [1, 0],
    [0, 1]
])

# Generate Gaussian samples
GAUSSIAN_SAMPLES = np.random.multivariate_normal(MEAN_VECTOR, COV_MATRIX, size=500)
X = GAUSSIAN_SAMPLES[:, 0]
Y = GAUSSIAN_SAMPLES[:, 1]
X, Y = np.meshgrid(X, Y)
XY = np.column_stack([X.flat, Y.flat])

# Use multivariate normal as the result for Z
Z = multivariate_normal.pdf(XY, mean=MEAN_VECTOR.tolist(), cov=COV_MATRIX.tolist())
Z = Z.reshape(X.shape)

# Linear transformation

# Plot
FIG = plt.figure()
AX = Axes3D(FIG)

SURF = AX.plot_surface(X, Y, Z, cmap=cm.winter)
FIG.colorbar(SURF, shrink=0.5, aspect=5)

plt.show()


