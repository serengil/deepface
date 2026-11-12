# built-in dependencies
from typing import List, Union, cast

# third-party dependencies
import numpy as np

# project dependencies
from deepface.config.minmax import get_minmax_values
from deepface.commons.embed_utils import is_flat_embedding


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

    def normalize_vector(vector: List[float]) -> List[float]:
        # Clip before scaling, including values outside the model's range.
        values = np.asarray(vector)
        if values.dtype == object:
            # Arbitrarily large Python integers need object storage; preserve
            # Python's NaN handling when clipping this uncommon input type.
            clipped = np.asarray([min(max(value, dim_min), dim_max) for value in vector])
        else:
            clipped = np.clip(values, dim_min, dim_max)
        return cast(List[float], ((clipped - dim_min) / (dim_max - dim_min)).tolist())

    if is_flat_embedding(embeddings):
        return normalize_vector(cast(List[float], embeddings))

    # Process each row separately to retain support for empty or unequal rows.
    return [normalize_vector(cast(List[float], emb)) for emb in embeddings]


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
        norm = float(np.linalg.norm(embeddings))
        if norm > 0:
            embeddings = cast(List[float], embeddings)  # let type checker know
            embeddings = (np.array(embeddings) / norm).tolist()
    else:
        normalized_embeddings = []
        for emb in embeddings:
            emb = cast(List[float], emb)  # let type checker know
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                normalized_emb = (np.array(emb) / norm).tolist()
            else:
                normalized_emb = emb
            normalized_embeddings.append(normalized_emb)
        embeddings = normalized_embeddings

    return embeddings
