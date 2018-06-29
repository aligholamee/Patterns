import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

colors = ['blue', 'red', 'green', 'cyan', 'purple']

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


# Plot points
class_1 = np.array([
    [1, 1],
    [-3, 1]
])

class_2 = np.array([
    [2, -1],
    [-3, -1]
])

# Plot decision lines
for i in range(5):
    lab = 'y = ' + str(weights[i][1]) + 'x1 + ' + str(weights[i][2]) + 'x2 + ' + str(weights[i][0])
    plt.plot(x, (-1 * weights[i][1] * x + weights[i][0] * x) / (weights[i][2]), colors[i], label=lab)

plt.legend(loc='bottm center')
plt.show()