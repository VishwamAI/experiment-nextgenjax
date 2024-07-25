# This __init__.py will mimic the structure of JAX's __init__.py to align with the user's request for a JAX-like structure in NextGenJax.

# Import core functionalities
from .aliases import nnp, npf, agb, alr, acl, aes, aok, ast, adt

# Alias the package for simplified importing
import nextgenjax as nnp

# Import specific functionalities for direct access
from nextgenjax import random, grad, jit, tree_map

# Define aliases for components to enable the requested import syntax
random = nnp.random
grad = nnp.grad
jit = nnp.jit
tree_map = nnp.tree_map
