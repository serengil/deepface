# built-in dependencies
from typing import Any, Dict, List, Union, Optional
from abc import ABC, abstractmethod


# pylint: disable=unnecessary-pass, too-many-positional-arguments
class Database(ABC):
    @abstractmethod
    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        """
        Initialize the database client with connection details or an existing connection.
        """
        pass

    @abstractmethod
    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure that the embeddings table or indexes exist in the database.
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not require storing embeddings index."
            " Because it handles indexing internally."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support ANN search natively."
        )

    def search_by_id(
        self,
        ids: Union[List[str], List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Search records by their IDs.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement search_by_id method."
            " Because search by vector returns metadata already."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not require storing embeddings index."
            " Because it handles indexing internally."
        )
