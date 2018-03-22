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

# Config bin size and intervals
bin_size = 0.2; min_edge = 1; max_edge = 5
N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
bin_list = np.linspace(min_edge, max_edge, Nplus1)


fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].hist(class_1, bins=bin_list, color='darkblue', label='Iris Setosa')
ax[0].legend(loc='upper right')
ax[1].hist(class_2, bins=bin_list, color='green', label='Iris Veriscolor')
ax[1].legend(loc='upper right')

plt.show()

