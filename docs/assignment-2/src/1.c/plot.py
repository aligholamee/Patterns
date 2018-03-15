import pylab as pl
import numpy as np
from matplotlib import cm

color_map=[cm.winter,cm.autumn]

D = 2

# Different Mean Vectors
M1 = np.array([0.0, 2.0])
M2 = np.array([3.0, 1.0])
M3 = np.array([1.0, 0.0])

# Same Covariance Matrix
C = np.array([
    [1.0, 0.0],
    [0.0, 1/3]
])

X, Y = np.mgrid[-2:2:100j, -2:2:100j]
points = np.c_[X.ravel(), Y.ravel()]

invC = np.linalg.inv(C)

v = points - M1
g1 = -0.5*np.sum(np.dot(v, invC) * v, axis=1) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C))
g1.shape = 100, 100

v = points - M2
g2 = -0.5*np.sum(np.dot(v, invC) * v, axis=1) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C))
g2.shape = 100, 100

v = points - M3
g3 = -0.5*np.sum(np.dot(v, invC) * v, axis=1) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C))
g3.shape = 100, 100

fig, axes = pl.subplots(1, 6, dpi=120, figsize=(15, 5))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()
for ax in axes.ravel():
    ax.set_aspect("equal")


ax1.pcolormesh(X, Y, g1, cmap=color_map[0])
ax1.set_title("W1")
ax2.pcolormesh(X, Y, g2, cmap=color_map[0])
ax2.set_title("W2")
ax3.pcolormesh(X, Y, g3, cmap=color_map[0])
ax3.set_title("W3")
ax4.pcolormesh(X, Y, g1 > g2, cmap=color_map[1])
ax4.set_title("W1 > W2")
ax5.pcolormesh(X, Y, g2 > g3, cmap=color_map[1])
ax5.set_title("W2 > W3")
ax6.pcolormesh(X, Y, g1 > g3, cmap=color_map[1])
ax6.set_title("W1 > W3")

pl.show()