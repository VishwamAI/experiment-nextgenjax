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

class Array:
    def __init__(self, data):
        self.data = original_np.asarray(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def __repr__(self):
        return f"Array(shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

# Custom implementation of multiply to ensure it behaves as expected
def multiply(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    print("Entering multiply function")
    print(f"x1 type: {type(x1)}, x2 type: {type(x2)}")
    print(f"x1 shape: {x1.shape if hasattr(x1, 'shape') else 'scalar'}")
    print(f"x2 shape: {x2.shape if hasattr(x2, 'shape') else 'scalar'}")
    print(f"dtype: {dtype}")

    # Handle array-array and array-scalar multiplication
    if isinstance(x1, Array) and isinstance(x2, (int, float)):
        result = Array(x1.data * x2)
    elif isinstance(x2, Array) and isinstance(x1, (int, float)):
        result = Array(x2.data * x1)
    elif isinstance(x1, Array) and isinstance(x2, Array):
        if x1.shape != x2.shape:
            raise ValueError(f"Cannot multiply arrays with shapes {x1.shape} and {x2.shape}")
        result = Array(x1.data * x2.data)
    else:
        # Fall back to original numpy multiply for other cases
        result = original_np.multiply(x1, x2, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
        result = Array(result)

    print("Multiplication result:")
    print(f"Result type: {type(result)}")
    print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
    print(f"Result sample: {result.data[:2, :2, :2] if hasattr(result, 'shape') and len(result.shape) == 3 else result.data}")

    # If 'out' is provided, write the result to 'out' array
    if out is not None:
        out[...] = result.data
        print("Result written to 'out' parameter:", out)

    return result.data if isinstance(result, Array) else result

class NextGenJaxNumpy:
    def __init__(self):
        pass

    def multiply(self, x1, x2):
        result = multiply(x1, x2)
        return result.data if isinstance(result, Array) else result

__all__ = ['NextGenJaxNumpy', 'Array', 'multiply']
