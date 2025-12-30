# built-in dependencies
from typing import Any, Dict, List
from abc import ABC, abstractmethod


# pylint: disable=unnecessary-pass, too-many-positional-arguments
class Database(ABC):
    @abstractmethod
    def ensure_embeddings_table(self) -> None:
        """
        Ensure that the embeddings table or indexes exist in the database.
        """
        pass

    @abstractmethod
    def upsert_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        index_data: bytes,
    ) -> None:
        """
        Insert or update the embeddings index in the database.
        """
        pass

    @abstractmethod
    def get_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> bytes:
        """
        Retrieve the embeddings index from the database.
        """
        pass

    @abstractmethod
    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert embeddings into the database in batches.
        """
        pass

    @abstractmethod
    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all embeddings from the database in batches.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the database connection.
        """
        pass

    @abstractmethod
    def search_by_vector(
        self,
        vector: List[float],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        aligned: bool = True,
        l2_normalized: bool = False,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        ANN search using the main vector (embedding).
        """
        pass
