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

print(GAUSSIAN_SAMPLES)