import numpy as np
import matplotlib.pyplot as plt

def plotLines(w0, w1, w2):
    d_x = np.linspace(-5, 5, 1000)
    return 

weights = np.array([
    [0, 0, 0], 
    [1, 1, 1],
    [0, -1, 2],
    [-1, 2, 3],
    [1, -4, 3]
]) 

x = np.linspace(-5, 5, 1000)

for i in range(5):
    plt.plot(x, (-1 * weights[1] * x + weights[0] * x) / (weights[2] * x))

plt.show()