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

x = np.linspace(-4, 4, 1000)


# Plot points
class_1 = np.array([
    [1, 1],
    [-3, 1]
])

class_2 = np.array([
    [2, -1],
    [-3, -1]
])


plt.scatter(class_1[:, 0], class_1[0, :], marker='o', label='Class 1', c='darkred')

plt.scatter(2, -1, marker='o', label='Class 2', c='darkblue')
plt.scatter(-3, -1, marker='o', c='darkblue')
# plt.scatter(class_2[:, 0], class_2[0, :], marker='o', label='Class 2', c='darkblue')

# Plot decision lines
for i in range(0, 5):
    lab = 'y = ' + str(weights[i][1]) + 'x1 + ' + str(weights[i][2]) + 'x2 + ' + str(weights[i][0])
    plt.plot(x, (-1 * weights[i][1] * x + weights[i][0]) / (weights[i][2]), colors[i], label=lab)

plt.legend(loc='best')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()