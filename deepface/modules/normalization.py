# built-in dependencies
from typing import List, Union, cast

# third-party dependencies
import numpy as np

# project dependencies
from deepface.config.minmax import get_minmax_values


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


def normalize_embedding_minmax(
    model_name: str, embeddings: Union[List[float], List[List[float]]]
) -> Union[List[float], List[List[float]]]:
    """
    Normalize embeddings using min-max normalization based on model-specific min-max values.
    Args:
        model_name (str): Name of the model to get min-max values for.
        embeddings (List[float] or List[List[float]]): Embeddings to normalize.
    Returns:
        List[float] or List[List[float]]: Normalized embeddings.
    """
    dim_min, dim_max = get_minmax_values(model_name)

    if dim_max - dim_min == 0:
        return embeddings

    if is_flat_embedding(embeddings):
        embeddings = cast(List[float], embeddings)  # let type checker know

        # Clamp vals to [dim_min, dim_max] to ensure the norm-embedding stays in [0, 1]
        embeddings = [max(x, dim_min) for x in embeddings]  # lower-bound clamp
        embeddings = [min(x, dim_max) for x in embeddings]  # upper-bound clamp

        embeddings = [(x - dim_min) / (dim_max - dim_min) for x in embeddings]

    else:
        normalized_embeddings = []
        for emb in embeddings:
            emb = cast(List[float], emb)  # let type checker know

            # Clamp vals to [dim_min, dim_max] to ensure the norm-embedding stays in [0, 1]
            emb = [max(x, dim_min) for x in emb]  # lower-bound clamp
            emb = [min(x, dim_max) for x in emb]  # upper-bound clamp

            emb = [(min(max(x, dim_min), dim_max) - dim_min) / (dim_max - dim_min) for x in emb]
            normalized_embeddings.append(emb)
        embeddings = normalized_embeddings

    return embeddings


def normalize_embedding_l2(
    embeddings: Union[List[float], List[List[float]]],
) -> Union[List[float], List[List[float]]]:
    """
    Normalize embeddings using L2 normalization.
    Args:
        embeddings (List[float] or List[List[float]]): Embeddings to normalize.
    Returns:
        List[float] or List[List[float]]: L2-normalized embeddings.
    """
    if is_flat_embedding(embeddings):
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            embeddings = cast(List[float], embeddings)  # let type checker know
            norm = cast(float, norm)
            embeddings = (np.array(embeddings) / norm).tolist()
    else:
        normalized_embeddings = []
        for emb in embeddings:
            emb = cast(List[float], emb)  # let type checker know
            norm = np.linalg.norm(emb)
            if norm > 0:
                norm = cast(float, norm)
                normalized_emb = (np.array(emb) / norm).tolist()
            else:
                normalized_emb = emb
            normalized_embeddings.append(normalized_emb)
        embeddings = normalized_embeddings

    return embeddings
