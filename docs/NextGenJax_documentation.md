# NextGenJax Model Documentation

## Custom 'jit' Implementation

In the NextGenJax model, we have implemented a custom 'jit' function that serves as a decorator for just-in-time compilation of functions. This custom implementation is designed to replace the direct import from JAX and align with the custom framework being built for NextGenJax.

### Usage

To use the custom 'jit' decorator, simply import it from the `nextgenjax_model` module and apply it to your function definitions:

```python
from nextgenjax_model import custom_jit as jit

@jit
def your_function(args):
    # Function implementation
    pass
```

The custom 'jit' decorator will handle the tracing and compilation of the function, optimizing its performance for execution.

### Implementation Details

The custom 'jit' function currently includes placeholders for tracing and compilation logic. It is designed to be extended in the future with more advanced functionality as the NextGenJax model evolves.
