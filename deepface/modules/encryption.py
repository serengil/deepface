# built-in dependencies
from typing import List, Union, Optional, cast

# third-party dependencies
from lightphe import LightPHE
from lightphe.models.Tensor import EncryptedTensor
import numpy as np

# project dependencies
from deepface.commons.embed_utils import is_flat_embedding
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=no-else-return
def encrypt_embeddings(
    embeddings: Union[List[float], List[List[float]]], cryptosystem: Optional[LightPHE] = None
) -> Union[EncryptedTensor, List[EncryptedTensor], None]:
    """
    Encrypt embeddings using a provided cryptosystem.
    Args:
        embeddings (List[float] or List[List[float]]): Embeddings to encrypt.
        cryptosystem (LightPHE): Cryptosystem to use for encryption.
    Returns:
        EncryptedTensor or List[EncryptedTensor] or None: Encrypted embeddings or None
            if no cryptosystem is provided.
    """
    if cryptosystem is None:
        return None

    if is_flat_embedding(embeddings):
        embedding = cast(List[float], embeddings)  # let type checker know
        encrypted_embedding = encrypt_embedding(embedding, cryptosystem)
        return encrypted_embedding
    else:
        encrypted_embeddings: List[EncryptedTensor] = []
        embeddings = cast(List[List[float]], embeddings)
        for embedding in embeddings:
            encrypted_embedding = encrypt_embedding(embedding, cryptosystem)
            encrypted_embeddings.append(encrypted_embedding)

        if all(item is None for item in encrypted_embeddings):
            return None

        return encrypted_embeddings


def encrypt_embedding(embeddings: List[float], cryptosystem: LightPHE) -> Optional[EncryptedTensor]:
    """
    Encrypt an embedding using a provided cryptosystem.
    Args:
        embeddings (List[float]): Embedding to encrypt.
        cryptosystem (LightPHE): Cryptosystem to use for encryption.
    Returns:
        EncryptedTensor or None: Encrypted embedding or None if encryption is skipped.
    """
    if any(x < 0 for x in embeddings):
        logger.warn(
            "Skipping encryption because it contains negative values."
            "Consider to set minmax_normalize=True in DeepFace.represent method."
        )
        return None

    norm = np.linalg.norm(embeddings)
    if not np.isclose(norm, 1.0):
        logger.warn(
            "Skipping encryption because given embedding is not l_2 normalized."
            "Consider to set l2_normalize=True in DeepFace.represent method."
        )
        return None

    encrypted_embeddings = cryptosystem.encrypt(embeddings, silent=True)
    return encrypted_embeddings
