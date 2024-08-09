import unittest
from typing import Sequence, Any
import sys

if sys.platform != "win32":
    from jax import jit
    Shape = Sequence[int | Any]
else:
    jit = None
    Shape = None

class TestJitFunctionality(unittest.TestCase):

    def test_jit_decorator(self):
        @jit
        def square(x):
            return x * x

        self.assertEqual(square(2), 4)

if __name__ == '__main__':
    unittest.main()
