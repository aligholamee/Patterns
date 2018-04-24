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
def sample_count_in_bins(samples, bin_size):
    """
        Find the number of existing samples in each bin of the Histogram and return k * n / v as the density of that bin.
        Return a dictionary containing the bin steps and the density inside them.
    """

    # Estimation dictionary
    sample_counts = {
            'bin_1': 0,
            'bin_2': 0
    }

    for sample in samples:
        # Find the location of sample in Histogram
        bin_index = int(sample / bin_size) + 1
        bin_number_str = 'bin_' + str(bin_index)

        if bin_number_str in sample_counts:
            # Update the value of that key
            sample_counts[bin_number_str] += 1
        else:
            # Simply create that key inside dictionary and assign it as 1
            sample_counts[bin_number_str] = 1

    
    # Return the results dictionary
    return sample_counts



# One dimensional array of data
samples_1d = truncated_normal(MEAN, STANDARD_DEVIATION, NUM_SAMPLES, RANGE_MIN, RANGE_MAX)

# Estimate the density of the samples
sample_counts_dict = sample_count_in_bins(samples_1d, BIN_SIZE)

# Test the results
print(sample_counts_dict)
