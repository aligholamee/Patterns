import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X = np.random.standard_normal(100)
Y = np.zeros(X.shape)
Z = np.zeros(X.shape)

# Simulate the 1st density function
for i, value in enumerate(X):
    if(X[i] >= 0) and (X[i] <= 1):
        Y[i] = 2*X[i]

# Simulate the 2nd density function
for i, value in enumerate(X):
    if(X[i] >= 0) and (X[i] <= 1):
        Z[i] = 2*X[i] - 2

sns.distplot(Y, fit_kws={"color":"windows blue"})
sns.distplot(Z, fit_kws={"color":"faded green"})

plt.show()