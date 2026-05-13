# project dependencies
from deepface.modules.vectorstores.base import VectorStore
from deepface.modules.vectorstores.factory import create_vector_store

__all__ = ['VectorStore', 'create_vector_store']