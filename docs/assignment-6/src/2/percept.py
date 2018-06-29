import numpy as np
import matplotlib.pyplot as plt
import random

num_epoch = 50
learning_rate = 0.1

aug_norm_or_d = np.array([
    [-1, 0, 0, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1]
])

or_d = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1] 
])

aug_norm_xor_d = np.array([
    [-1, 0, 0, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [-1, -1, -1, 0]
])

xor_d = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# Generate random weights
weights = np.array([random.uniform(0, 1) for i in range(3)]).reshape(1, 3)

for i in range(num_epoch):
    # Load the whole dataset
    for (b, x, y, l) in np.ndenumerate(aug_norm_or_d):

        # Classify the loaded point using weights in step i - 1
        output = weights[1] * x + weights[2] * y + weights[0] * b
        if(output <= 0):
            # Update the weights
            weights[0] = weights[0] + learning_rate * b
            weights[1] = weights[1] + learning_rate * x
            weights[2] = weights[2] + learning_rate * y
        else:
            weights[0] = weights[0]
            weights[1] = weights[1]
            weights[2] = weights[2]
        




