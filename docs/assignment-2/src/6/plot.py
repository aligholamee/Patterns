import numpy as np
import matplotlib.pyplot as plt

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

