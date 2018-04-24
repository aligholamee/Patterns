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

# Implements saomple counting strategy
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


# Draws the density in matplotlib
def draw_density(range_min, range_max, which_bin, bin_size, num_samples_in_bin, hist_height_of_each_sample):

    interval_low = range_min + which_bin * bin_size
    interval_high = interval_low + bin_size

    # GENERATE MANY POINTS!!!!
    x = np.linspace(1, 21, 1000)
    plt.plot(x, list(map(lambda x: num_samples_in_bin*hist_height_of_each_sample if interval_low <= x <= interval_high else 0, x)), color='darkblue')


# Implemenets the density estimation method
def find_density(sample_count_dict, num_samples, bin_size):

    # This is the height of density for each sample that can be calculated as heigh = 1 / (n * v)
    # v is bin size in this case
    # n is the number of all samples

    height_of_density_for_each_sample = 1 / (num_samples * bin_size)

    # Iterate the sample_count_dict
    for bin_number, sample_count in sample_count_dict.iteritems():
        draw_density(RANGE_MIN, RANGE_MAX, bin_number, BIN_SIZE, sample_count, height_of_density_for_each_sample)


# One dimensional array of data
samples_1d = truncated_normal(MEAN, STANDARD_DEVIATION, NUM_SAMPLES, RANGE_MIN, RANGE_MAX)

# Estimate the density of the samples
sample_counts_dict = sample_count_in_bins(samples_1d, BIN_SIZE)

# plot the generated samples
plt.scatter(samples_1d, np.zeros_like(samples_1d), color='b')
plt.ylim(-0.1, 5)
plt.xlim(1, 20)
plt.show()