from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np

# Number of samples
NUM_SAMPLES = 100

# Bin size of the Histogram
BIN_SIZE = 2

# Known density parameters
MEAN = 8
STANDARD_DEVIATION = 5
RANGE_MIN = 1
RANGE_MAX = 25

# Generates random normal numbers in a range
def truncated_normal(mean, stddev, min, max):

    return truncnorm(
        (min - mean) / stddev, (max - mean) / stddev, loc=mean, scale=stddev
    )

# Implements the histogram density estimation methods
def estimate_histogram_density(samples, bin_size):
    """
        Find the number of existing samples in each bin of the Histogram and return k * n / v as the density of that bin.
    """


# One dimensional array of data
samples_1d = truncated_normal(MEAN, STANDARD_DEVIATION, RANGE_MIN, RANGE_MAX)

print(samples_1d)