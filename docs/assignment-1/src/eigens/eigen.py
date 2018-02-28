# ========================================
# [] File Name : eigen.py
#
# [] Creation Date : February 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Computing the eigenvectors and eigenvalues of a random matrix.
"""
import numpy as np
import numpy.linalg as linalg

S_MATRIX = [
    [4, 0, 0],
    [0, 2, 2],
    [0, 9, -5]
]

VALUES, VECTORS = linalg.eig(S_MATRIX)

print("Eigenvalues: ", VALUES)
print("Eigenvectors: ", VECTORS)
