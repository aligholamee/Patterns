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

color_map=[cm.winter,cm.autumn]

MEAN_VECTOR = np.array([0,0])

COV_MATRIX = np.array([
    [1, 0],
    [0, 1]
])

# Transformation configuration
A_T = np.array([
    [-5, 5],
    [1, 1]
])

B_T = np.array([0.5, 1])

# Linear transformation
def linear_map(transformation_matrix, bias_matrix, data):
    '''
        Performs a matrix multiplication as a linear map
        also translates the new mean vector and covariance matrix
    '''
    Y = np.matmul(transformation_matrix, np.transpose(data)) + bias_matrix

    return Y

# Finds the new mean and covariance matrix
def transform_cov_mean(prev_mean, prev_cov, transformation_matrix, bias_matrix):
    '''
        Find the new mean as A*prev_mean + B
        Find the new cov as A^(T) prev_cov A
    '''
    return np.transpose(np.matmul(transformation_matrix, np.transpose(prev_mean))) + bias_matrix, np.matmul(np.matmul(np.transpose(transformation_matrix), prev_cov), transformation_matrix)

# Function to plot each distribution
def plot_dists(samples, mean_vec, cov_mat, i):
    '''
        This will plot normal surface for each sample set
    '''

    x = samples[:, 0]
    y = samples[:, 1]
    x, y = np.meshgrid(x, y)
    xy = np.column_stack([x.flat, y.flat])

    # Use multivariate normal as the result for Z
    z = multivariate_normal.pdf(xy, mean=mean_vec.tolist(), cov=cov_mat.tolist())
    z = z.reshape(x.shape)

    # Plot
    FIG = plt.figure()
    AX = Axes3D(FIG)

    SURF = AX.plot_surface(x, y, z, cmap=color_map[i])
    FIG.colorbar(SURF, shrink=0.5, aspect=5)

    plt.show()

# Generate Gaussian samples and plot the results
GAUSSIAN_SAMPLES = np.random.multivariate_normal(MEAN_VECTOR, COV_MATRIX, size=500)
plot_dists(GAUSSIAN_SAMPLES, MEAN_VECTOR, COV_MATRIX, 1)

# Transform the Gaussian samples
TRANSFORMED_SAMPLES = np.array([])
for point in GAUSSIAN_SAMPLES:
    TRANSFORMED_SAMPLES = np.append(TRANSFORMED_SAMPLES, linear_map(data=point, transformation_matrix=A_T, bias_matrix=B_T))
    
# Reshape the transformed samples like GAUSSIAN SAMPLES
TRANSFORMED_SAMPLES = TRANSFORMED_SAMPLES.reshape((500, 2))

# Plot the transformed samples
new_mean, new_cov = transform_cov_mean(MEAN_VECTOR, COV_MATRIX, A_T, B_T)
plot_dists(TRANSFORMED_SAMPLES, new_mean, new_cov, 0)




