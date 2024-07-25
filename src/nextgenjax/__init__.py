# This __init__.py will mimic the structure of JAX's __init__.py to align with the user's request for a JAX-like structure in NextGenJax.

# Import core functionalities
import nextgenjax as nnp
import nextgenjax.numpy as jnp
from nextgenjax import random, grad, jit, tree_map

# Import components with aliases to reflect their purpose and maintain the JAX-like structure
from .components.AIPhoenix_DistributedTrainer import AIPhoenix_DistributedTrainer as distributed
from .components.AIPhoenix_SpeechTranscriber import AIPhoenix_SpeechTranscriber as speech
from .components.AIPhoenix_OptimizerKit import AIPhoenix_OptimizerKit as optimizer
from .components.AIPhoenix_EnvSimulator import AIPhoenix_EnvSimulator as env
from .components.AIPhoenix_ChainedLM import AIPhoenix_ChainedLM as lang
from .components.AIPhoenix_GraphBuilder import AIPhoenix_GraphBuilder as graph
from .components.AIPhoenix_NeuralFramework import AIPhoenix_NeuralFramework as neural

# Alias the package for simplified importing
import nextgenjax as nnp

# Import specific functionalities for direct access
from nextgenjax import random, grad, jit, tree_map

# Define aliases for components to enable the requested import syntax
random = optimizer.random
grad = graph.grad
jit = neural.jit
tree_map = lang.tree_map
