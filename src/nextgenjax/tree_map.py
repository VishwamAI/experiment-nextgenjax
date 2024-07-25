# Custom tree_map module for NextGenJax

def tree_map(func, tree):
    """
    A custom implementation of the tree mapping function.
    This function applies a function to each leaf in a nested structure.
    """
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(func, elem) for elem in tree)
    elif isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    else:
        return func(tree)
