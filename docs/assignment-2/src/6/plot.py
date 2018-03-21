import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class_1[:, 0], class_1[:, 1], color='darkblue')
ax.scatter(class_2[:, 0], class_2[:, 1], color='red')

# Customize the figure background color
plt.gca().patch.set_facecolor('white')
ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
plt.show()

