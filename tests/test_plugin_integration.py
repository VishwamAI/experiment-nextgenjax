import sys
print("Starting test_plugin_integration.py")
print("Python path:", sys.path)

import jax.numpy as jnp
import jax.random as jrandom
import optax
import chex
print("Attempting to import haiku...")
import haiku as hk
print("Haiku imported successfully")
print("Haiku path:", hk.__file__)
print("Haiku attributes:", dir(hk))

# Instantiate the model
# model = NextGenJaxModel()

# Create a test tensor
key = jrandom.PRNGKey(0)
test_tensor = jrandom.uniform(key, shape=(1, 10, 10, 10, 3))

# Test if the model can handle JAX arrays
# processed_tensor = model.process_input(test_tensor, test_tensor)
# print('Processed tensor shape:', processed_tensor.shape)

# Test optimization using optax
optimizer = optax.adam(learning_rate=1e-3)
# opt_state = optimizer.init(model.get_params())

# Test type checking with chex
# chex.assert_shape(processed_tensor, (1, 10, 10, 10, model.num_classes))

# Test neural network creation with Haiku
print("Haiku attributes before transform:", dir(hk))
def forward_pass(x):
    return hk.nets.MLP([64, 32, 10])(x)  # Replace model.num_classes with a fixed value

print("Attempting to use hk.transform...")
transformed_forward = hk.transform(forward_pass)
print("hk.transform successful")
params = transformed_forward.init(key, jnp.zeros((1, 10)))
output = transformed_forward.apply(params, key, test_tensor.reshape(-1, 10))
print('Haiku MLP output shape:', output.shape)
