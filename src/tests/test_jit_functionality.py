import unittest
from src.nextgenjax.jit.jit import jit

class TestJitFunctionality(unittest.TestCase):

    def test_jit_decorator(self):
        @jit
        def square(x):
            return x * x

        self.assertEqual(square(2), 4)

if __name__ == '__main__':
    unittest.main()
