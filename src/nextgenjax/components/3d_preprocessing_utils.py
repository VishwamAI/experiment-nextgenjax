from nextgenjax.custom_numpy import NextGenJaxNumpy as np, multiply
import time

def load_3d_data(filepath):
    # Implement logic to load 3D data from a file
    data = np.load(filepath)
    return data

def normalize_3d_data(data):
    # Implement logic to normalize 3D data
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data

def augment_3d_data(data, augmentation_params):
    # Implement logic to augment 3D data
    # Example: Apply random rotations
    augmented_data = np.rot90(data, k=augmentation_params.get('rotations', 1))
    return augmented_data

def transform_3d_data(data, transformation_params):
    print("Entering transform_3d_data function")
    print(f"Input data type: {type(data)}")
    print(f"Input data shape: {data.shape}")
    print(f"Transformation params: {transformation_params}")
    print(f"multiply function: {multiply}")

    scale_factor = float(transformation_params.get('scale', 1.0))
    print(f"Input data shape: {data.shape}")
    print(f"Scale factor: {scale_factor}")

    # Remove unnecessary type conversions, as these should be handled by the custom multiply function
    print("Original data type:", type(data))
    print("Original data dtype:", getattr(data, 'dtype', 'N/A'))
    print("Sample of original data:", data[:2, :2, :2] if hasattr(data, '__getitem__') else data)

    print("Data after conversion to float32:", data[:2, :2, :2])

    print("Data shape:", data.shape)
    print("Data type:", data.dtype)
    print("Scale factor:", scale_factor)
    print("Sample of original data:", data[:2, :2, :2])

    print("Type of data before scaling:", type(data))
    print("Type of scale_factor:", type(scale_factor))

    # Apply scaling using custom multiply function with fallback
    print("Before multiplication:")
    print("data:", data)
    print("scale_factor:", scale_factor)
    print("About to call multiply function")
    print(f"scale_factor: {scale_factor}")
    print(f"data type before multiply: {type(data)}")

    print("Attempting to use custom multiply function")
    print("Before multiply call:")
    print(f"Data type: {type(data)}")
    print(f"Data shape: {data.shape}")
    print(f"Data sample: {data[:2, :2, :2]}")
    print(f"Scale factor: {scale_factor}")
    try:
        start_time = time.time()
        transformed_data = multiply(data, scale_factor, dtype=np.float32)
        execution_time = time.time() - start_time

        print(f"Custom multiply execution time: {execution_time} seconds")

        if execution_time > 5:  # Adjust this threshold as needed
            raise TimeoutError("Custom multiply function took too long")

        print("After multiply call:")
        print(f"Transformed data type: {type(transformed_data)}")
        print(f"Transformed data shape: {transformed_data.shape}")
        print(f"Transformed data sample: {transformed_data[:2, :2, :2]}")
    except Exception as e:
        print(f"Custom multiply failed with error: {str(e)}")
        print(f"Exception details: {e}")
        print("Falling back to standard numpy multiplication")
        transformed_data = data * scale_factor
        print("After scaling (fallback):")
        print(f"Transformed data type: {type(transformed_data)}")
        print(f"Transformed data shape: {transformed_data.shape}")
        print(f"Transformed data sample: {transformed_data[:2, :2, :2]}")

    print("Type of transformed_data:", type(transformed_data))
    print("Sample of transformed data:", transformed_data[:2, :2, :2])

    print("Data immediately after scaling:", transformed_data[:2, :2, :2])

    print("Transformed data shape:", transformed_data.shape)
    print("Transformed data type:", transformed_data.dtype)
    print("Sample of transformed data:", transformed_data[:2, :2, :2])

    # Check if the transformed data is different from the original data
    if np.array_equal(data, transformed_data):
        print("Warning: multiplication did not change the data.")
        print("Original data mean:", np.mean(data))
        print("Transformed data mean:", np.mean(transformed_data))
    elif np.allclose(data, transformed_data, rtol=1e-5, atol=1e-8):
        print("Warning: The transformed data is very close to the original data. Scaling might not have occurred as expected.")
        print("Original data mean:", np.mean(data))
        print("Transformed data mean:", np.mean(transformed_data))
    else:
        print("Scaling applied successfully.")
        print("Original data mean:", np.mean(data))
        print("Transformed data mean:", np.mean(transformed_data))

    print("Final transformed data:", transformed_data[:2, :2, :2])
    print("Is transformed data a np.Array?", isinstance(transformed_data, np.Array))
    print("Transformed data dtype:", transformed_data.dtype)

    print(f"Final transformed data shape: {transformed_data.shape}")
    print(f"Final transformed data sample: {transformed_data[:2, :2, :2]}")

    print("Exiting transform_3d_data function")
    return transformed_data
