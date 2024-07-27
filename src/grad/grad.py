# Custom grad module for NextGenJax

def grad(func):
    """
    A custom implementation of the gradient function.
    This function will compute the gradient of a given function with respect to its parameters.
    """
    def grad_func(*args, **kwargs):
        # Compute the gradient of 'func' with respect to the first argument
        # This is a simplified example and does not represent the full complexity of automatic differentiation
        # In practice, this would involve complex backpropagation logic
        # Here we assume that 'func' returns a scalar output and args[0] is a vector of parameters
        gradients = []  # Placeholder for gradients
        params = args[0]
        for i in range(len(params)):
            # Numerical gradient approximation using finite differences
            # This is not efficient or accurate for real use cases
            h = 1e-4  # Small perturbation
            params_perturbed = params.copy()
            params_perturbed[i] += h
            grad_approx = (func(params_perturbed, *args[1:], **kwargs) - func(params, *args[1:], **kwargs)) / h
            gradients.append(grad_approx)
        return gradients
    
    return grad_func
