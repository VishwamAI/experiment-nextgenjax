from nextgenjax_model import custom_jit as jit

@jit
def test_function(x):
    return x * x

print(test_function(2))
