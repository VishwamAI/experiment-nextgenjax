# AIPhoenix_OptimizerKit.py
import nextgenjax
import nextgenjax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple

class AIPhoenix_OptimizerKit:
    @staticmethod
    def clip_by_global_norm(max_norm: float) -> Callable:
        def transform(updates: Any) -> Any:
            g_norm = nextgenjax.tree_util.tree_map(jnp.linalg.norm, updates)
            g_norm = jnp.sqrt(jnp.sum(jnp.square(g_norm)))
            factor = jnp.minimum(1.0, max_norm / (g_norm + 1e-6))
            return nextgenjax.tree_util.tree_map(lambda t: t * factor, updates)
        return transform

    @staticmethod
    def scale_by_adam(b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> Callable:
        def init_fn(params: Any) -> Tuple[Any, Any]:
            m = nextgenjax.tree_util.tree_map(jnp.zeros_like, params)
            v = nextgenjax.tree_util.tree_map(jnp.zeros_like, params)
            return m, v

        def update_fn(updates: Any, state: Tuple[Any, Any], params: Any = None) -> Tuple[Any, Tuple[Any, Any]]:
            m, v = state
            m = nextgenjax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, updates)
            v = nextgenjax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * (g ** 2), v, updates)
            updates = nextgenjax.tree_util.tree_map(lambda m, v: m / (jnp.sqrt(v) + eps), m, v)
            return updates, (m, v)

        return init_fn, update_fn

    @staticmethod
    def sgd(learning_rate: float) -> Callable:
        def init_fn(params: Any) -> None:
            return None

        def update_fn(updates: Any, state: None, params: Any = None) -> Tuple[Any, None]:
            return nextgenjax.tree_util.tree_map(lambda g: -learning_rate * g, updates), state

        return init_fn, update_fn

    @staticmethod
    def rmsprop(learning_rate: float, decay: float = 0.9, eps: float = 1e-8) -> Callable:
        def init_fn(params: Any) -> Any:
            return nextgenjax.tree_util.tree_map(jnp.zeros_like, params)

        def update_fn(updates: Any, state: Any, params: Any = None) -> Tuple[Any, Any]:
            new_state = nextgenjax.tree_util.tree_map(
                lambda s, g: decay * s + (1 - decay) * (g ** 2),
                state, updates)
            updates = nextgenjax.tree_util.tree_map(
                lambda g, s: g / (jnp.sqrt(s + eps)),
                updates, new_state)
            return updates, new_state

        return init_fn, update_fn

    @staticmethod
    def adaptive_learning_rate(initial_lr: float, decay_rate: float, decay_steps: int) -> Callable:
        def schedule(step: int) -> float:
            return initial_lr * (decay_rate ** (step // decay_steps))
        return schedule

    @staticmethod
    def distributed_optimization(optimizer: Callable, num_devices: int) -> Callable:
        def distributed_init(params: Any) -> Any:
            return nextgenjax.pmap(optimizer[0])(params)

        def distributed_update(updates: Any, state: Any, params: Any = None) -> Tuple[Any, Any]:
            return nextgenjax.pmap(optimizer[1])(updates, state, params)

        return distributed_init, distributed_update

    @staticmethod
    @nextgenjax.jit
    def custom_gradient_transformation(grads: Any) -> Any:
        return nextgenjax.tree_util.tree_map(lambda g: g - nextgenjax.lax.pmean(g, axis_name='i'), grads)

# Utility functions to mimic JAX API
def random():
    return nextgenjax.random

def grad(fun: Callable, argnums: int = 0, has_aux: bool = False) -> Callable:
    return nextgenjax.grad(fun, argnums=argnums, has_aux=has_aux)

def jit(fun: Callable, static_argnums: Optional[Any] = None) -> Callable:
    return nextgenjax.jit(fun, static_argnums=static_argnums)

def tree_map(f: Callable, tree: Any) -> Any:
    return nextgenjax.tree_util.tree_map(f, tree)
