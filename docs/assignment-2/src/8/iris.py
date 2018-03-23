import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, pi, exp, log, e
from sklearn.metrics import confusion_matrix

DATA_ROOT = './data/'
COL_NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# Univariate normal density
def u_n_d(x, mean, variance):
    '''
        Return the univariate density of given set x
    '''
    return 1/(sqrt(2*pi*variance)) * exp(-((x - mean)**2 / (2*variance)))
 
# Finds the Baysian decision boundary
def b_d_b(x, mean1, mean2, var1, var2, prior1, prior2):
    '''
        Return a point on the decision boundary
    '''
    return (log(var2/var1, e) - (x - mean1)**2/(2*var1) + (x - mean2)**2/(2*var2) - log(prior2/prior1, e))

# Find the accuracy of the classification
def find_accuracy(actual, predicted):
    '''
        Divide the number of truly predicted labels to the total number of labels
    '''
    return (100.0 * np.sum(actual['class'] == predicted['class']) / actual.size)

    
# Read the train and test data
df = pd.read_fwf(DATA_ROOT+'Iris_train.dat')
df2 = pd.read_fwf(DATA_ROOT+'Iris_test.dat')

# Rename the columns
df.columns = COL_NAMES
df2.columns = COL_NAMES

# ax = plt.Figure()
# ax.add_subplot(111)

train_data = df[['sepal-width', 'class']]
test_data = df2[['sepal-width', 'class']]

# plt.hist(pd)
class_1 = df.loc[train_data['class'] == 0.0]
class_2 = df.loc[train_data['class'] == 1.0]

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
ax[0].set_xlabel('sample')
ax[0].set_ylabel('sepal-width')
ax[1].hist(class_2, bins=bin_list, color='green', label='Iris Veriscolor')
ax[1].legend(loc='upper right')
ax[1].set_xlabel('sample')
ax[1].set_ylabel('sepal-width')

# plt.show()

# # Class Prior probabilities
prior_1 = class_1.size/df.size
prior_2 = class_2.size/df.size
print("Prior probability of Setosa: ", prior_1)
print("Prior probability of Veriscolor: ", prior_2)

# Plot the density functions of two classes
ax2 = plt.subplot()
x_range = np.linspace(1, 6, 100)

yy1 = [u_n_d(x, class_1_mean, class_1_variance) for x in x_range]
yy2 = [u_n_d(x, class_2_mean, class_2_variance) for x in x_range]
yy3 = [b_d_b(x, class_1_mean, class_2_mean, class_1_variance, class_2_variance, prior_1, prior_2) for x in x_range]

ax2.plot(x_range, yy1, label='Setosa Density', color='darkblue')
ax2.plot(x_range, yy2, label='Veriscolor Density', color='green')
ax2.set_xlabel('sepal-width')
ax2.set_ylabel('probability density')

# Plot the Bayesian decision boundary
ax2.plot(x_range, yy3, label='Decision Boundary', color='red')
ax2.set_ylim(0, 1.5)
ax2.legend(loc='upper right')

# plt.show()

# Classify the training data  
cols = ['sepal-width', 'class']
train_prediction = pd.DataFrame(columns=cols)

for index, x in train_data.iterrows():
    cond = b_d_b(x['sepal-width'], class_1_mean, class_2_mean, class_1_variance, class_2_variance, prior_1, prior_2)
    if(cond >= 0):
        # Classify as class 1
        train_prediction = train_prediction.append({'sepal-width': x['sepal-width'], 'class': 0.0}, ignore_index=True)
    else:
        # Classify as class 2
        train_prediction = train_prediction.append({'sepal-width': x['sepal-width'], 'class': 1.0}, ignore_index=True)

# Classify the test data  
test_prediction = pd.DataFrame(columns=cols)

for index, x in test_data.iterrows():
    cond = b_d_b(x['sepal-width'], class_1_mean, class_2_mean, class_1_variance, class_2_variance, prior_1, prior_2)
    if(cond >= 0):
        # Classify as class 1
        test_prediction = test_prediction.append({'sepal-width': x['sepal-width'], 'class': 0.0}, ignore_index=True)
    else:
        # Classify as class 2
        test_prediction = test_prediction.append({'sepal-width': x['sepal-width'], 'class': 1.0}, ignore_index=True)

# Calculate the accuracy of precition
train_acc = find_accuracy(train_data, train_prediction)
test_acc = find_accuracy(test_data, test_prediction)
print("Training Accuracy: ", train_acc)
print("Test Accuracy: ", test_acc)
print("Training Error: ", 1.0 - train_acc)
print("Test Error: ", 1.0 - test_acc)
print("Training Confusion Matrix: ")
print(confusion_matrix(train_data['class'], train_prediction['class']))
print("Testing Confusion Matrix: ")
print(confusion_matrix(test_data['class'], test_prediction['class']))