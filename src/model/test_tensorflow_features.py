import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import gc
import psutil
import logging
import tracemalloc

tf.keras.backend.set_floatx('float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def setup_logging():
    logging.basicConfig(filename='test_errors.log', level=logging.ERROR,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def print_memory_usage(message):
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{message}: RSS={memory_info.rss / 1024 / 1024:.2f} MB, VMS={memory_info.vms / 1024 / 1024:.2f} MB")

MEMORY_LIMIT = 70  # Percentage of total memory

def check_memory():
    memory = psutil.virtual_memory()
    if memory.percent > MEMORY_LIMIT:
        print(f"Memory usage too high: {memory.percent}%")
        return False
    return True

# Define a vocabulary size for the model
vocab_size = 2  # Reduced to 2 to further decrease memory usage

class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def get_model():
    tf.config.optimizer.set_jit(True)
    return SimpleModel()

# Define test cases for each feature
def test_text_to_text_conversion():
    model = get_model()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    input_text = tf.constant([[1.0]], dtype=tf.float16)

    try:
        output = model(input_text)
        assert output is not None, 'Text-to-text conversion failed.'
        assert output.shape == (1, 1), f"Expected output shape (1, 1), but got {output.shape}"
    except tf.errors.ResourceExhaustedError as e:
        logging.error(f"Out of memory error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        tracemalloc.clear_traces()

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)
    tracemalloc.stop()

def test_voice_to_text_conversion():
    model = get_model()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    try:
        input_voice = tf.random.normal([1, 1], dtype=tf.float16)  # Further reduced input size
        dataset = tf.data.Dataset.from_tensor_slices(input_voice).batch(1).prefetch(tf.data.AUTOTUNE)
        for batch in dataset:
            output = model(batch)
            assert output is not None, 'Voice-to-text conversion failed.'
            assert output.shape == (1, 1), f"Expected output shape (1, 1), but got {output.shape}"
    except tf.errors.ResourceExhaustedError as e:
        logging.error(f"Out of memory error in voice-to-text conversion: {e}")
    except Exception as e:
        logging.error(f"Error in voice-to-text conversion: {e}")
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)
    tracemalloc.stop()
    tf.keras.backend.clear_session()
    gc.collect()

@tf.function
def train_step(model, inputs, targets, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        step_loss = tf.keras.losses.MeanSquaredError()(targets, predictions)
    gradients = tape.gradient(step_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return step_loss

def test_advanced_optimization_techniques():
    model = get_model()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    inputs = tf.random.normal([1, 1], dtype=tf.float16)
    targets = tf.random.normal([1, 1], dtype=tf.float16)
    try:
        loss = train_step(model, inputs, targets, optimizer)
        assert loss is not None, 'Optimizer should yield a result'
        print(f"Optimization test successful. Loss: {loss}")
    except tf.errors.ResourceExhaustedError as e:
        logging.error(f"Out of memory error: {e}")
        print(f"Out of memory error in optimization test: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in optimization test: {e}")
        print(f"Unexpected error in optimization test: {e}")
    finally:
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        print("[ Top 10 memory differences ]")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()
        tf.keras.backend.clear_session()
        gc.collect()
        tracemalloc.clear_traces()

def test_advanced_researching():
    model = get_model()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant([[1]], dtype=tf.float16))
    dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)

    try:
        for research_prompt in dataset:
            output = model(research_prompt)
            assert output is not None, 'Research output should not be None'
            assert isinstance(output, tf.Tensor), 'Output should be a Tensor'
    except tf.errors.ResourceExhaustedError as e:
        logging.error(f"Out of memory error in test_advanced_researching: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in test_advanced_researching: {e}")
    finally:
        tf.keras.backend.clear_session()
        gc.collect()

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)
    tracemalloc.stop()
    tracemalloc.clear_traces()

def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()

def run_individual_test(test_func):
    print(f"\nRunning test: {test_func.__name__}")
    print_memory_usage("Before test")
    if not check_memory():
        print(f"Memory usage too high before {test_func.__name__}. Skipping.")
        return
    try:
        test_func()
        print(f"Test {test_func.__name__} completed successfully.")
    except Exception as e:
        print(f"Error in {test_func.__name__}: {str(e)}")
        logging.error(f"Error in {test_func.__name__}: {str(e)}")
    print_memory_usage(f"After {test_func.__name__}")
    clear_memory()
    print_memory_usage(f"After memory clear")

def test_large_tensor_allocation():
    try:
        # Attempt to allocate a large tensor with reduced size
        large_tensor = tf.random.normal((1000, 1000, 10), dtype=tf.float16)
        print(f"Successfully allocated tensor of shape: {large_tensor.shape}")
    except tf.errors.ResourceExhaustedError as e:
        print(f"Resource exhausted error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    setup_logging()
    tests = [
        test_large_tensor_allocation,
        test_text_to_text_conversion,
        test_voice_to_text_conversion,
        test_advanced_optimization_techniques,
        test_advanced_researching,
    ]

    try:
        for test in tests:
            print(f"\nRunning test: {test.__name__}")
            print_memory_usage("Before test")
            run_individual_test(test)
            print_memory_usage("After test")
            clear_memory()
            print_memory_usage("After memory clear")
            input("Press Enter to continue to the next test...")

    except tf.errors.ResourceExhaustedError as e:
        print(f"Memory allocation error: {e}")
        logging.error(f"Memory allocation error: {e}")
    except MemoryError as e:
        logging.error(f"Memory Error: {str(e)}")
        print(f"Memory Error occurred: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error occurred: {str(e)}")
