import numpy as np
import matplotlib.pyplot as plt

# Grab the inv function as inverse + log and det
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

def decision_boundary(x_vec, meu_1, meu_2, cov_1, cov_2):
    g1 = x_vec * compute_W(inverse(cov_1)) * x_vec + compute_w(inverse(cov_1), meu_1) * x_vec + compute_w0(cov_1, inverse(cov_1), meu_1, 7/17) 
    g2 = x_vec * compute_W(inverse(cov_2)) * x_vec + compute_w(inverse(cov_2), meu_2) * x_vec + compute_w0(cov_2, inverse(cov_2), meu_2, 10/17)

    return g1 - g2
class_1 = np.array([
    [1.5, 0],
    [1, 1],
    [2, 1],
    [0.5, 2],
    [1.5, 2],
    [2, 3],
    [2.5, 2]
])

class_2 = np.array([
    [1.5, 0.5],
    [1.5, -0.5],
    [0.5, 0.5], 
    [0.5, -0.5],
    [1, -1],
    [-0.5, 0.5],
    [-1, -1],
    [-1.5, 0],
    [-1.5, 1],
    [-2, -1]
])

# Find mean and covariance of class 1
meu_1 = np.matrix(np.mean(class_1, axis=0)).T
cov_1 = np.cov(class_1.T)
print("Class 1 mean: ", meu_1)
print("Class 1 covariance: ", cov_1)

# Find mean and covariance of class 2
meu_2 = np.matrix(np.mean(class_2, axis=0)).T
cov_2 = np.cov(class_2.T)
print("Class 2 mean: ", meu_2)
print("Class 2 covariance: ", cov_2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(class_1[:, 0], class_1[:, 1], marker='o', color='darkblue')
ax.scatter(class_2[:, 0], class_2[:, 1], marker='^', color='red')
plt.legend(['Class1 (w1)', 'Class2 (w2)'], loc='upper left') 
plt.xlabel('x1')
plt.ylabel('x2')
ftext = 'p(x|c1) ~ N(mu1=[1.571,1.571], cov1=[[0.452, 0.119], [0.119, 0.952]])\np(x|c2) ~ N(mu2=[-0.15, -0.15], cov2=[[1.725, 0.002], [0.002, 0.558]])'
plt.figtext(.9,.9, ftext, fontsize=10, ha='right')
plt.show()

