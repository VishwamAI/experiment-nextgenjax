import numpy as np
from nextgenjax.components.three_d_preprocessing_utils import transform_3d_data

# Create a sample 3D array
sample_data = np.random.rand(10, 10, 10)
scale_factor = 2.0

# Call the transform_3d_data function
transformed_data = transform_3d_data(sample_data, scale_factor)

# Output the results
print("Original data sample:", sample_data[:2, :2, :2])
print("Transformed data sample:", transformed_data[:2, :2, :2])
