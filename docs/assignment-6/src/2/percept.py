import numpy as np
import matplotlib.pyplot as plt
import random

num_epoch = 50
learning_rate = 0.1

or_d = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

xor_d = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# Generate random weights
weights = np.array([random.uniform(0, 1) for i in range(3)]).reshape(1, 3)

def relu()
for i in range(num_epoch):
    # Load the whole dataset
    for (x, y, l) in np.ndenumerate(dataset):

        # Classify the loaded point using weights in step i - 1
        output = relu(weights[1] * x + weights[2] * y + weights[0])
        if(output != l):
            # Update the weights
            w[0] = 
            w[1] = 
            w[2] = 




