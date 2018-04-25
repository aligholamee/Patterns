# ========================================
# [] File Name : parzen_1d.py
#
# [] Creation Date : April 2018
#
# [] Author 2 : Ali Gholami
# ========================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


# Seed the random number generator
np.random.seed(1)
NUM_SAMPLES = 20

# Density characteristics
modal1_mean = 2
modal2_mean = 10
modal1_variance = 1
modal2_variance = 1

# Generate some samples from the given multi modal distribution
sample_set = np.concatenate(np.random.normal(modal1_mean, modal1_variance, int(0.3 * NUM_SAMPLES),
                            np.random.normal(modal2_mean, modal2_variance, int(0.7 * NUM_SAMPLES))))

