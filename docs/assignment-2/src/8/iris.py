import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_ROOT = './data/'
COL_NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# Read the train and test data
df = pd.read_fwf(DATA_ROOT+'Iris_train.dat')

# Rename the columns
df.columns = COL_NAMES

# ax = plt.Figure()
# ax.add_subplot(111)
# plt.hist(pd)
class_1 = df.loc[df['class'] == 0.0]
class_2 = df.loc[df['class'] == 1.0]

class_1 = class_1['sepal-width']
class_2 = class_2['sepal-width']

# Find the distributions
class_1_mean = np.mean(class_1)
class_2_mean = np.mean(class_2)

class_1_variance = np.var(class_1)
class_2_variance = np.var(class_2)

print("Setosa mean: ", class_1_mean)
print("Setosa variance: ", class_1_variance)
print("Veriscolor mean: ", class_2_mean)
print("Veriscolor variance: ", class_2_variance)

ax = plt.subplot()
ax.plot(class_1, marker='^', color='darkblue', label='Iris Setosa')
ax.plot(class_2, marker='o', color='green', label='Iris Veriscolor')
plt.legend(loc='upper right')

plt.show()

