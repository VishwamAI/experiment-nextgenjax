import torch

class CudaPlugin:
    def __init__(self):
        # Check for CUDA availability
        self.is_cuda_available = torch.cuda.is_available()

    def get_device(self):
        # Return the appropriate device
        return torch.device('cuda' if self.is_cuda_available else 'cpu')

    def to_device(self, tensor):
        # Move tensor to the device
        return tensor.to(self.get_device())
