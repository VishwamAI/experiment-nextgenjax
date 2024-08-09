import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "jax_unsupported: mark test to be skipped on platforms where JAX is not supported (e.g., Windows)"
    )
