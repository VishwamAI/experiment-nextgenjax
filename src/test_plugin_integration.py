
from nextgenjax.nextgenjax_model import NextGenJaxModel

# Instantiate the model
model = NextGenJaxModel()

# Create a test tensor
import torch
test_tensor = torch.rand(10, 10)

# Test if the model can move the tensor to the appropriate device
device_tensor = model.to_device(test_tensor)
print('Test tensor device:', device_tensor.device)

