# AIPhoenix_OptimizerKit.py
import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple
from jax.tree_util import tree_map, tree_leaves

class AIPhoenix_OptimizerKit:
    @staticmethod
    def clip_by_global_norm(max_norm: float) -> Callable:
        def clip_fn(grads: Any) -> Any:
            # Calculate the global norm of the gradients
            global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in tree_leaves(grads)))
            # Debug print statements
            print(f"Calculated global norm: {global_norm}")
            # Calculate the scale for clipping
            scale = jnp.minimum(max_norm / (global_norm + 1e-6), 1.0)
            # Debug print statements
            print(f"Calculated scale for clipping: {scale}")
            # Clip the gradients
            clipped_grads = tree_map(lambda g: g * scale, grads)
            # Debug print statements
            print(f"Clipped gradients: {clipped_grads}")
            return clipped_grads
        return clip_fn

    @staticmethod
    def sgd(learning_rate: float) -> Callable:
        def init_fn(params: Any) -> None:
            return None

        def update_fn(grads: Any, state: None, params: Any = None) -> Tuple[Any, None]:
            # Calculate the updates using SGD
            updates = tree_map(lambda g: -learning_rate * g, grads)
            # Debug print statements
            print(f"SGD updates: {updates}")
            return updates, state

        return init_fn, update_fn

    # ... rest of the AIPhoenix_OptimizerKit class ...
