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

# Linear transformation
def linear_map(transformation_matrix, bias_matrix, data, prior_mean, prior_cov):
    '''
        Performs a matrix multiplication as a linear map
        also translates the new mean vector and covariance matrix
    '''
    Y = np.matmul(transformation_matrix, data) + bias_matrix
    cov_y = np.matmul(np.transpose(transformation_matrix), prior_cov) + transformation_matrix
    mean_y = np.matmul(transformation_matrix, prior_mean) + bias_matrix

    return Y, cov_y, mean_y
    
# Generate Gaussian samples
GAUSSIAN_SAMPLES = np.random.multivariate_normal(MEAN_VECTOR, COV_MATRIX, size=500)
X = GAUSSIAN_SAMPLES[:, 0]
Y = GAUSSIAN_SAMPLES[:, 1]
X, Y = np.meshgrid(X, Y)
XY = np.column_stack([X.flat, Y.flat])

# Use multivariate normal as the result for Z
Z = multivariate_normal.pdf(XY, mean=MEAN_VECTOR.tolist(), cov=COV_MATRIX.tolist())
Z = Z.reshape(X.shape)



# Plot
FIG = plt.figure()
AX = Axes3D(FIG)

SURF = AX.plot_surface(X, Y, Z, cmap=cm.winter)
FIG.colorbar(SURF, shrink=0.5, aspect=5)

plt.show()


