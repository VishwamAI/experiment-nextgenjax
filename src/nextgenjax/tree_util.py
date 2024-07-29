# tree_util module for NextGenJax

def tree_map(func, *trees):
    """
    Apply a function to each element in a nested structure.
    """
    if not trees:
        return None
    if len(trees) == 1:
        tree = trees[0]
        if isinstance(tree, (list, tuple)):
            return type(tree)(tree_map(func, t) for t in tree)
        elif isinstance(tree, dict):
            return {k: tree_map(func, v) for k, v in tree.items()}
        else:
            return func(tree)
    else:
        first_tree = trees[0]
        if isinstance(first_tree, (list, tuple)):
            return type(first_tree)(tree_map(func, *(t[i] for t in trees)) for i in range(len(first_tree)))
        elif isinstance(first_tree, dict):
            return {k: tree_map(func, *(t[k] for t in trees)) for k in first_tree}
        else:
            return func(*trees)
