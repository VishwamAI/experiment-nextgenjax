import unittest
import numpy as np
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from nextgenjax.components.learning_networking_mechanisms import LearningMechanism, NetworkingMechanism
from nextgenjax.components.reinforcement_learning_module import ReinforcementLearningModule
from nextgenjax.components.building_mechanisms import BuildingMechanism
from nextgenjax.components.benchmarks import Benchmarks
from nextgenjax.components.three_d_preprocessing_utils import load_3d_data, normalize_3d_data, augment_3d_data, transform_3d_data
import jax.numpy as jnp
import jax.random as jrandom
import optax
import chex
import haiku as hk

class Test3DFunctionalities(unittest.TestCase):

    def test_load_3d_data(self):
        # Test loading of 3D data
        filepath = 'path/to/3d/data.npy'
        data = load_3d_data(filepath)
        self.assertIsInstance(data, np.ndarray)

    def test_normalize_3d_data(self):
        # Test normalization of 3D data
        data = np.random.rand(10, 10, 10)
        normalized_data = normalize_3d_data(data)
        self.assertTrue(np.all((normalized_data >= 0) & (normalized_data <= 1)))

    def test_augment_3d_data(self):
        # Test augmentation of 3D data
        data = np.random.rand(10, 10, 10)
        augmented_data = augment_3d_data(data, {'rotations': 2})
        self.assertEqual(augmented_data.shape, data.shape)

    def test_transform_3d_data(self):
        # Test transformation of 3D data
        data = np.random.rand(10, 10, 10)
        print("Original data shape:", data.shape)
        print("Original data sample:", data[0, 0, :5])

        transformed_data = transform_3d_data(data, {'scale': 2.0})
        print("Transformed data shape:", transformed_data.shape)
        print("Transformed data sample:", transformed_data[0, 0, :5])

        print("Expected data sample:", (data * 2.0)[0, 0, :5])
        print("Are arrays equal?", np.allclose(transformed_data, data * 2.0, rtol=1e-5, atol=1e-8))

        self.assertTrue(np.all(np.isclose(transformed_data, data * 2.0, rtol=1e-5, atol=1e-8)))

    def test_learning_mechanism(self):
        # Test learning mechanism processing
        learning_mechanism = LearningMechanism()
        data = np.random.rand(10, 10)
        processed_data = learning_mechanism.process(data)
        self.assertTrue(np.all((processed_data >= 0) & (processed_data <= 1)))

    def test_networking_mechanism(self):
        # Test networking mechanism interaction
        networking_mechanism = NetworkingMechanism([LearningMechanism()])
        data = np.random.rand(10, 10, 10)  # Create sample 3D data
        results = networking_mechanism.interact(data)
        self.assertIsInstance(results, list)

    def test_reinforcement_learning_module(self):
        # Test reinforcement learning module training and evaluation
        rl_module = ReinforcementLearningModule()
        environment = MockEnvironment()
        agent = MockAgent()
        rl_module.train(environment, agent, 10)
        total_reward = rl_module.evaluate(environment, agent)
        self.assertIsInstance(total_reward, float)

    def test_building_mechanism(self):
        # Test building mechanism for generating 3D models
        building_mechanism = BuildingMechanism()
        model = building_mechanism.generate_3d_model('text input')
        self.assertIsInstance(model, np.ndarray)

    def test_benchmarks(self):
        # Test benchmarks for evaluating 3D functionalities
        benchmarks = Benchmarks()
        model = MockModel()
        data = np.random.rand(10, 10, 10)
        results = benchmarks.evaluate_3d_functionalities(model, data)
        metrics = benchmarks.performance_metrics(results)
        self.assertIsInstance(metrics, dict)

# Mock classes for testing purposes
class MockEnvironment:
    def reset(self):
        return np.zeros((10, 10, 10))

    def step(self, action):
        return (np.zeros((10, 10, 10)), 0.0, True, {})

class MockAgent:
    def act(self, state):
        return 0

    def learn(self, state, action, reward, next_state, done):
        pass

class MockModel:
    def process(self, data):
        return np.random.rand(10, 10, 10)

if __name__ == '__main__':
    unittest.main()

class Test3DFunctionalities(unittest.TestCase):
    # ... (previous test methods)

    def test_multiply_function(self):
        from nextgenjax.custom_numpy import NextGenJaxNumpy
        import numpy as np

        # Create a sample 3D array
        data = np.random.rand(5, 5, 5)
        scale = 2.0

        # Create an instance of NextGenJaxNumpy
        ngj_np = NextGenJaxNumpy()

        # Use the multiply function
        result = ngj_np.multiply(data, scale)

        # Print debug information
        print("Original data sample:", data[0, 0, :5])
        print("Multiplied data sample:", result[0, 0, :5])
        print("Expected data sample:", (data * scale)[0, 0, :5])

        # Check if the multiplication was performed correctly
        self.assertTrue(np.allclose(result, data * scale, rtol=1e-5, atol=1e-8))
