import numpy as np
import matplotlib.pyplot as plt

# Grab the inv function as inverse
inverse = np.linalg.inv
log = np.math.log 
det = np.linalg.det

# Functions to compute the weights
def compute_W(cov_inverse):
    return -(1/2) * cov_inverse

def compute_w(cov_inverse, mean_vector):
    return cov_inverse * mean_vector

def compute_w0(cov, cov_inverse, mean_vector, prior_prob):
    return -(1/2) * mean_vector.T * cov_inverse * mean_vector - (1/2) * log(det(cov)) + log(prior_prob)

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

# Find covariance matrix, inverse and mean
mean_vec1 = np.matrix(np.mean(set_1, axis=0)).T
mean_vec2 = np.matrix(np.mean(set_2, axis=0)).T
cov_set1 = np.cov(set_1.T)
cov_set2 = np.cov(set_2.T)
cov_set1_inverse = inverse(cov_set1)
cov_set2_inverse = inverse(cov_set2)

print("First class mean vector:\n", mean_vec1)
print("Second class mean vector:\n", mean_vec2)

print("First class covariance matrix:\n", cov_set1)
print("Second class covariance matrix:\n", cov_set2)

# Compute the weights of the 1'st discriminator
W1 = compute_W(cov_set1_inverse)
w1 = compute_w(cov_set1_inverse, mean_vec1)
w10 = compute_w0(cov_set1, cov_set1_inverse, mean_vec1, 1/2)

# Compute the weights of the 2'nd discriminator
W2 = compute_W(cov_set2_inverse)
w2 = compute_w(cov_set2_inverse, mean_vec2)
w20 = compute_w0(cov_set2, cov_set2_inverse, mean_vec2, 1/2)

# Print the results
print("The results for the first discriminator: ")
print(W1)
print(w1)
print(w10)

print("The results for the second discriminator: ")
print(W2)
print(w2)
print(w20)
