# Custom numpy module for NextGenJax, aliased as 'jnp' to mimic JAX's numpy extension

import numpy as original_np

# Here we would define custom wrappers or extensions to the numpy library functions
# that are required by the NextGenJax model, similar to how JAX extends numpy.

# For now, this module will directly map to the original numpy functions.
# This is a placeholder and should be extended with custom functionality as needed.

# Expose all numpy attributes to be available under this module
from numpy import *

# Alias the entire numpy library for convenience
jnp = original_np
