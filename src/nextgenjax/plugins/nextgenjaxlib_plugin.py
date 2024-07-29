from .cuda_plugin import CudaPlugin

class NextGenJaxLib:
    def __init__(self):
        # Initialize the CUDA plugin
        self.cuda_plugin = CudaPlugin()

    def to_device(self, tensor):
        # Use the CUDA plugin to move tensor to the device
        return self.cuda_plugin.to_device(tensor)
