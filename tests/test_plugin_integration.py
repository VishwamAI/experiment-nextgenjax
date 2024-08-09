import sys
import pytest
from typing import Sequence, Any
print("Starting test_plugin_integration.py")
print("Python path:", sys.path)

if sys.platform != "win32":
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import optax
    import chex
    import haiku as hk
    Shape = Sequence[int | Any]
else:
    jax = jnp = jrandom = optax = chex = hk = None
    Shape = None

# Mark all tests in this file as jax_unsupported
pytestmark = pytest.mark.jax_unsupported

@pytest.mark.skipif(sys.platform == "win32", reason="JAX not supported on Windows")
def test_jax_integration():
    assert jax is not None, "JAX should be imported on non-Windows platforms"
    print("JAX version:", jax.__version__)

    # Create a test tensor
    key = jrandom.PRNGKey(0)
    test_tensor = jrandom.uniform(key, shape=(1, 10, 10, 10, 3))

    # Test optimization using optax
    optimizer = optax.adam(learning_rate=1e-3)
    assert optimizer is not None, "Optax optimizer should be created"

    # Test type checking with chex
    chex.assert_shape(test_tensor, (1, 10, 10, 10, 3))

    # Test neural network creation with Haiku
    def forward_pass(x):
        return hk.nets.MLP([64, 32, 10])(x)

    transformed_forward = hk.without_apply_rng(hk.transform(forward_pass))
    params = transformed_forward.init(jrandom.PRNGKey(0), jnp.zeros((1, 10)))
    output = transformed_forward.apply(params, test_tensor.reshape(-1, 10))
    print('Haiku MLP output shape:', output.shape)
    assert output.shape == (1000, 10), "Unexpected output shape from Haiku MLP"

if __name__ == "__main__":
    if sys.platform != "win32":
        test_jax_integration()
    else:
        print("Skipping JAX integration test on Windows")
