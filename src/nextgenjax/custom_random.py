# Custom random module for NextGenJax

import numpy as np

class Random:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def normal(self, key, shape, mean=0.0, std=1.0):
        # Mimic JAX's normal distribution function
        return self.rng.normal(loc=mean, scale=std, size=shape)

    def randint(self, key, minval, maxval, shape=None):
        # Mimic JAX's randint function
        return self.rng.integers(low=minval, high=maxval, size=shape)

    def PRNGKey(self, seed):
        # Mimic JAX's PRNGKey function
        return np.random.SeedSequence(seed)

# Instantiate the Random class for use in NextGenJax
random = Random()
