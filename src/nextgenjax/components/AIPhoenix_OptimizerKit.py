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
        def optimizer(params: Any, grads: Any) -> Any:
            # Calculate the updates using SGD
            updates = tree_map(lambda p, g: p - learning_rate * g, params, grads)
            # Debug print statements
            print(f"SGD updates: {updates}")
            return updates

        return optimizer

    @staticmethod
    def scale_by_adam(learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> Callable:
        def init_fn(params: Any) -> Dict[str, Any]:
            m = tree_map(jnp.zeros_like, params)
            v = tree_map(jnp.zeros_like, params)
            return {'m': m, 'v': v, 't': jnp.array(0)}

        def update_fn(grads: Any, state: Dict[str, Any], params: Any = None) -> Tuple[Any, Dict[str, Any]]:
            m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state['m'], grads)
            v = tree_map(lambda v, g: b2 * v + (1 - b2) * jnp.square(g), state['v'], grads)
            t = state['t'] + 1
            m_hat = tree_map(lambda m: m / (1 - b1 ** t), m)
            v_hat = tree_map(lambda v: v / (1 - b2 ** t), v)
            updates = tree_map(lambda m, v, g: -learning_rate * m / (jnp.sqrt(v) + eps), m_hat, v_hat, grads)
            new_state = {'m': m, 'v': v, 't': t}
            return updates, new_state

        return init_fn, update_fn

    @staticmethod
    def custom_gradient_transformation(grad_fn: Callable) -> Callable:
        def transformation(grads: Any) -> Any:
            transformed_grads = grad_fn(grads)
            updates = tree_map(lambda g: -g, transformed_grads)
            return updates

        return transformation

    @staticmethod
    def rmsprop(learning_rate: float = 0.01, decay: float = 0.9, eps: float = 1e-8) -> Callable:
        def init_fn(params: Any) -> Dict[str, Any]:
            return tree_map(jnp.zeros_like, params)

        def update_fn(grads: Any, state: Dict[str, Any], params: Any = None) -> Tuple[Any, Dict[str, Any]]:
            new_state = tree_map(lambda s, g: decay * s + (1 - decay) * jnp.square(g), state, grads)
            updates = tree_map(lambda g, s: -learning_rate * g / (jnp.sqrt(s) + eps), grads, new_state)
            return updates, new_state

        return init_fn, update_fn
