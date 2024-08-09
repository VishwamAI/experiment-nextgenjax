import unittest
from typing import Sequence, Any
from src.nextgenjax.components.AIPhoenix_OptimizerKit import AIPhoenix_OptimizerKit
from src.nextgenjax.grad.grad import grad
from src.nextgenjax.jit.jit import jit
from src.nextgenjax import tree_map
import random
import jax
import jax.numpy as jnp
import jax.random as jrandom

Shape = Sequence[int | Any]

class TestAIPhoenixOptimizerKit(unittest.TestCase):

    def test_imports(self):
        # Test if the custom imports work as intended
        self.assertIsNotNone(random)
        self.assertIsNotNone(grad)
        self.assertIsNotNone(jit)
        self.assertIsNotNone(tree_map)

    def test_gradient_descent(self):
        # Test the gradient descent optimizer
        optimizer = AIPhoenix_OptimizerKit.sgd(learning_rate=0.1)
        params = jnp.array([1.0, 2.0, 3.0])
        grads = jnp.array([0.1, 0.1, 0.1])
        print("Before optimization - params:", params, "grads:", grads)
        updated_params = optimizer(params, grads)
        print("After optimization - updated_params:", updated_params)
        expected_params = params - 0.1 * grads
        self.assertTrue(jnp.allclose(expected_params, updated_params))

    def test_adam_optimizer(self):
        # Test the Adam optimizer
        optimizer = AIPhoenix_OptimizerKit.scale_by_adam()
        params = jnp.array([1.0, 2.0, 3.0])
        grads = jnp.array([0.1, 0.1, 0.1])
        init_state = optimizer[0](params)
        updates, new_state = optimizer[1](grads, init_state, params)
        self.assertIsNotNone(updates)
        self.assertIsNotNone(new_state)

    def test_rmsprop_optimizer(self):
        # Test the RMSProp optimizer
        optimizer = AIPhoenix_OptimizerKit.rmsprop(learning_rate=0.1)
        params = jnp.array([1.0, 2.0, 3.0])
        grads = jnp.array([0.1, 0.1, 0.1])
        init_state = optimizer[0](params)
        updates, new_state = optimizer[1](grads, init_state)
        self.assertIsNotNone(updates)
        self.assertIsNotNone(new_state)

    def test_clip_by_global_norm(self):
        # Test the clip_by_global_norm function
        clip_fn = AIPhoenix_OptimizerKit.clip_by_global_norm(max_norm=1.0)
        grads = jax.tree_util.tree_map(lambda x: jnp.array([x, x]), [0.5, 1.5, 2.5])
        clipped_grads = clip_fn(grads)
        norms = jax.tree_util.tree_map(jnp.linalg.norm, clipped_grads)
        for norm in norms:
            self.assertLessEqual(norm, 1.0)

    def test_custom_gradient_transformation(self):
        # Test the custom gradient transformation
        grad_fn = lambda x: jnp.array([x, x])
        print("Before transformation - grad_fn type:", type(grad_fn))
        print("Before transformation - grad_fn value:", grad_fn)
        transformed_grad_fn = AIPhoenix_OptimizerKit.custom_gradient_transformation(grad_fn)
        print("After transformation - transformed_grad_fn type:", type(transformed_grad_fn))
        print("After transformation - transformed_grad_fn value:", transformed_grad_fn)
        grads = transformed_grad_fn(jnp.array([1.0, 2.0, 3.0]))
        print("After applying transformed_grad_fn - grads type:", type(grads))
        print("After applying transformed_grad_fn - grads value:", grads)
        self.assertIsNotNone(grads)

if __name__ == '__main__':
    unittest.main()
