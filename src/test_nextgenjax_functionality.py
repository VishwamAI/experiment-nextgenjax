from src.nextgenjax_model import NextGenJaxModel, nnp
from src.nextgenjax.grad.grad import grad
from src.nextgenjax.jit.jit import jit
from src.nextgenjax.pmap.pmap import pmap
import scipy as sp
import matplotlib.pyplot as plt
import jaxlib

# Instantiate the model
model = NextGenJaxModel()

# Create a dummy input tensor for testing the 3D convolutional layers
input_tensor_3d = nnp.random.normal(loc=0.0, scale=1.0, size=(1, 64, 64, 64, 3))
# Create a dummy input tensor for testing the 2D convolutional layers
input_tensor_2d = nnp.random.normal(loc=0.0, scale=1.0, size=(1, 1, 64, 64, 3))

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

if output is not None:
    print("Output shape:", output.shape)
else:
    print("Output is None")

# Test the grad module by defining a simple loss function and computing its gradient
def loss_fn(params, inputs_2d, inputs_3d, targets):
    predictions = model.process_input(inputs_2d, inputs_3d)
    return nnp.mean((predictions - targets) ** 2)

# Create dummy targets tensor for testing the grad module
targets_tensor = nnp.random.normal(loc=0.0, scale=1.0, size=(1, model.num_classes))

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
