# built-in dependencies
from typing import Dict, Any, Optional

# project dependencies
from deepface.modules.vectorstores.base import VectorStore
from deepface.modules.vectorstores.local_backend import LocalVectorStore
from deepface.modules.vectorstores.milvus_backend import MilvusVectorStore
from deepface.commons.logger import Logger

logger = Logger()


def create_vector_store(store_type: str, config: Optional[Dict[str, Any]] = None) -> VectorStore:
    """
    Factory function to create vector store instances.

    Args:
        store_type: Type of vector store ('local', 'milvus', etc.)
        config: Configuration dictionary for the store

    Returns:
        VectorStore instance
    """
    store_type = store_type.lower()

    if store_type == 'local':
        return LocalVectorStore(config)
    elif store_type == 'milvus':
        return MilvusVectorStore(config)
    else:
        available_stores = ['local', 'milvus']
        raise ValueError(f"Unsupported vector store type: {store_type}. Available: {available_stores}")