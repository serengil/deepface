# built-in dependencies
from typing import Dict, Any, Optional

# project dependencies
from deepface.modules.verification_backends.base import VerificationBackend
from deepface.modules.verification_backends.local_backend import LocalVerificationBackend
from deepface.modules.verification_backends.vector_backend import VectorVerificationBackend
from deepface.commons.logger import Logger

logger = Logger()


def create_verification_backend(backend_type: str, config: Optional[Dict[str, Any]] = None) -> VerificationBackend:
    """
    Factory function to create verification backend instances.

    Args:
        backend_type: Type of verification backend ('local', 'vector', etc.)
        config: Configuration dictionary for the backend

    Returns:
        VerificationBackend instance
    """
    backend_type = backend_type.lower()

    if backend_type == 'local':
        return LocalVerificationBackend(config)
    elif backend_type in ('vector', 'milvus', 'pinecone', 'weaviate', 'pgvector'):
        return VectorVerificationBackend(config)
    else:
        available = ['local', 'vector', 'milvus', 'pinecone', 'weaviate', 'pgvector']
        raise ValueError(f"Unsupported verification backend: {backend_type}. Available: {available}")