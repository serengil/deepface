# built-in dependencies
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np

# project dependencies
from deepface.commons.logger import Logger

logger = Logger()


class VectorStore(ABC):
    """
    Abstract base class for vector storage and retrieval backends.
    Provides a pluggable interface for storing and searching face embeddings.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the vector store.

        Args:
            config (dict): Configuration parameters for the backend.
        """
        self.config = config or {}

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the backend.
        """
        pass

    @abstractmethod
    def register(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> List[int]:
        """
        Register embeddings with associated metadata.

        Args:
            embeddings: List of embedding vectors.
            metadata: List of metadata dictionaries (e.g., img_name, facial_area).
            model_name: Name of the model used for embeddings.
            detector_backend: Detector backend used.
            align: Whether alignment was performed.
            l2_normalize: Whether L2 normalization was applied.

        Returns:
            List of IDs assigned to the registered embeddings.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        distance_metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: The query embedding vector.
            model_name: Name of the model used for embeddings.
            detector_backend: Detector backend used.
            align: Whether alignment was performed.
            l2_normalize: Whether L2 normalization was applied.
            k: Number of top results to return.
            threshold: Distance threshold for filtering results.
            distance_metric: Distance metric ('cosine', 'euclidean', etc.).

        Returns:
            List of dictionaries with search results, each containing
            'id', 'distance', 'metadata', etc.
        """
        pass

    @abstractmethod
    def update(
        self,
        id: int,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: str = "",
        detector_backend: str = "",
        align: bool = True,
        l2_normalize: bool = False,
    ) -> bool:
        """
        Update an embedding and/or its metadata.

        Args:
            id: ID of the embedding to update.
            embedding: New embedding vector (optional).
            metadata: New metadata (optional).
            model_name: Name of the model used for embeddings.
            detector_backend: Detector backend used.
            align: Whether alignment was performed.
            l2_normalize: Whether L2 normalization was applied.

        Returns:
            True if update was successful.
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: List[int],
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> bool:
        """
        Delete embeddings by IDs.

        Args:
            ids: List of embedding IDs to delete.
            model_name: Name of the model used for embeddings.
            detector_backend: Detector backend used.
            align: Whether alignment was performed.
            l2_normalize: Whether L2 normalization was applied.

        Returns:
            True if deletion was successful.
        """
        pass

    @abstractmethod
    def count(
        self,
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> int:
        """
        Count the number of embeddings for given parameters.

        Args:
            model_name: Name of the model used for embeddings.
            detector_backend: Detector backend used.
            align: Whether alignment was performed.
            l2_normalize: Whether L2 normalization was applied.

        Returns:
            Number of embeddings.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close any connections or resources.
        """
        pass