import unittest

# Initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Discover and add tests to the test suite
suite.addTests(loader.discover(start_dir='.', pattern='test_*.py'))

# Run the test suite
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
