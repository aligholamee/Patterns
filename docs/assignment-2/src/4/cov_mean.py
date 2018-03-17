import numpy as np

set_1 = np.array([
    [0, 0],
    [0, 1],
    [2, 2],
    [3, 1],
    [3, 2],
    [3, 3]
])

set_2 = np.array([
    [6, 9],
    [8, 9],
    [9, 8],
    [9, 9],
    [9, 10],
    [8, 11]
])

print("First class mean vector:\n", set_1.mean(axis=0))
print("Second class mean vector:\n", set_2.mean(axis=0))

print("First class covariance matrix:\n", np.cov(set_1.T))
print("Second class covariance matrix:\n", np.cov(set_2.T))

