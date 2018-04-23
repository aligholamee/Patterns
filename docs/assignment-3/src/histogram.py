import matplotlib.pyplot as plt
import numpy as np

# Number of samples
NUM_SAMPLES = 100

# Bin size of the Histogram
BIN_SIZE = 2

# Known density parameters
MEAN = 5
STANDARD_DEVIATION = 3
RANGE_MIN = 1
RANGE_MAX = 20

# Generates random normal numbers in a range
def truncated_normal(mean, std, num_samples, min, max):
    """
        Return samples with normal distribution inside the given region
    """
    return  (np.random.normal(loc=mean, scale=std, size=num_samples) % (max - min) + min)

# Implements the histogram density estimation methods
def estimate_histogram_density(samples, bin_size):
    """
        Find the number of existing samples in each bin of the Histogram and return k * n / v as the density of that bin.
        Return a dictionary containing the bin steps and the density inside them.
    """

    # Estimation dictionary
    estimation_dict = {
            'bin_1': 0,
            'bin_2': 0
    }

    for sample in samples:
        # Find the location of sample in Histogram
        bin_index = sample / bin_size + 1
        bin_number_str = 'bin_' + str(bin_index)

        # In case the bin item doesn't exist
        # Simply create that key inside dictionary
        if(!estimation_dict[bin_number_str])
            estimation_dict[bin_number_str] = 0

        
        estimation_dict[bin_number_str] = 



# One dimensional array of data
samples_1d = truncated_normal(MEAN, STANDARD_DEVIATION, NUM_SAMPLES, RANGE_MIN, RANGE_MAX)

# Estimate the density of the samples
density_estimate_dict = estimate_histogram_density(samples_1d)
