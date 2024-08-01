import sys
print("Starting test_nextgenjax_functionality.py")
print("Python path:", sys.path)

from jax import grad, jit, pmap
import scipy as sp
import matplotlib.pyplot as plt
import jaxlib
import jax.numpy as jnp
import jax.random as jrandom
import optax
import chex
import haiku as hk

print("Attempting to import NextGenJaxModel...")
from nextgenjax.nextgenjax_model import NextGenJaxModel
print("NextGenJaxModel imported successfully")
print("NextGenJaxModel path:", NextGenJaxModel.__file__)

# Instantiate the model
model = NextGenJaxModel()

# Create a dummy input tensor for testing the 3D convolutional layers
key = jrandom.PRNGKey(0)
input_tensor_3d = jrandom.normal(key, shape=(1, 64, 64, 64, 3))
# Create a dummy input tensor for testing the 2D convolutional layers
key, subkey = jrandom.split(key)
input_tensor_2d = jrandom.normal(subkey, shape=(1, 1, 64, 64, 3))

# Print input shapes and model information
print("Input tensor 2D shape:", input_tensor_2d.shape)
print("Input tensor 3D shape:", input_tensor_3d.shape)
print("Model num_classes:", model.num_classes)

# Pass the input tensors through the model to test the convolutional layers
try:
    print("Calling process_input method...")
    output = model.process_input(input_tensor_2d, input_tensor_3d)
    print("process_input method completed successfully")
except Exception as e:
    print(f"Error in process_input method: {str(e)}")
    raise

# Add debug print statements
print("Debug: About to check 'output'")
print("Debug: Type of 'output':", type(output) if 'output' in locals() else "output is not defined")
print("Debug: Value of 'output':", output if 'output' in locals() else "output is not defined")

# Implement model processing and output validation
if output is not None:
    print("Output shape:", output.shape)
    chex.assert_shape(output, (1, model.num_classes))
else:
    print("Output is None")

# Test the grad module by defining a simple loss function and computing its gradient
def loss_fn(params, inputs_2d, inputs_3d, targets):
    predictions = model.process_input(inputs_2d, inputs_3d)
    return jnp.mean((predictions - targets) ** 2)

# Create dummy targets tensor for testing the grad module
key, subkey = jrandom.split(key)
targets_tensor = jrandom.normal(subkey, shape=(1, model.num_classes))

# Compute the gradients of the loss function with respect to the model parameters
gradients = jit(grad(loss_fn))(model.get_params(), input_tensor_2d, input_tensor_3d, targets_tensor)

# Print the gradients to verify the grad module functionality
print('Gradients:', gradients)

# Verify scipy integration by performing a simple operation
scipy_result = sp.fft.fft([0, 1, 0, 1])
print('SciPy FFT result:', scipy_result)

# Verify matplotlib integration by plotting a simple graph
plt.plot([0, 1, 0, 1])
plt.title('Matplotlib Integration Test')
plt.show()

# Verify XLA integration by checking the version of jaxlib
print('XLA (jaxlib) version:', jaxlib.__version__)

# Test optimization using optax
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(model.get_params())

# Test neural network creation with Haiku
def forward_pass(x):
    return hk.nets.MLP([64, 32, model.num_classes])(x)

transformed_forward = hk.transform(forward_pass)
params = transformed_forward.init(key, jnp.zeros((1, 10)))

# Add print statements to debug the reshaping operation
print("Shape of input_tensor_3d:", input_tensor_3d.shape)
print("Total elements in input_tensor_3d:", jnp.prod(jnp.array(input_tensor_3d.shape)))

# Calculate the correct first dimension for reshaping
total_elements = jnp.prod(jnp.array(input_tensor_3d.shape))
new_first_dim = total_elements // model.num_classes
intended_shape = (new_first_dim, model.num_classes)

# Ensure the total number of elements remains the same
if total_elements % model.num_classes != 0:
    raise ValueError("The total number of elements is not divisible by model.num_classes, cannot reshape properly.")

print("Intended reshape:", intended_shape)
print("Total elements after reshape:", jnp.prod(jnp.array(intended_shape)))

output = transformed_forward.apply(params, key, input_tensor_3d.reshape(intended_shape))
print('Haiku MLP output shape:', output.shape)
