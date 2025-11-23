from typing import List, Union


def is_flat_embedding(x: Union[List[float], List[List[float]]]) -> bool:
    """
    Check if the embeddings represent a single flat list of floats
        rather than a list of list of float.
    Args:
        x (List[float] or List[List[float]]): Embeddings to check.
    Returns:
        bool: True if x is a flat list of floats, False otherwise.
    """
    return isinstance(x, list) and all(isinstance(i, (int, float)) for i in x)
