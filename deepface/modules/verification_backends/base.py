# built-in dependencies
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import numpy as np

# project dependencies
from deepface.commons.logger import Logger

logger = Logger()


class VerificationBackend(ABC):
    """
    Abstract base class for verification backends.
    Provides a pluggable interface for identity verification using different storage strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize verification backend.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the backend."""
        pass

    @abstractmethod
    def register(
        self,
        img_path: Union[str, np.ndarray, IO[bytes], List[float]],
        identity: str,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        align: bool = True,
        normalization: str = "base",
        l2_normalize: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Register an identity with face embedding.

        Args:
            img_path: Image path, numpy array, or pre-calculated embedding
            identity: Unique identity identifier
            model_name: Face recognition model
            detector_backend: Face detector
            align: Whether to align face
            normalization: Normalization method
            l2_normalize: L2 normalization flag

        Returns:
            dict with registration result
        """
        pass

    @abstractmethod
    def verify(
        self,
        img1_path: Union[str, np.ndarray, IO[bytes], List[float]],
        img2_path: Optional[Union[str, np.ndarray, IO[bytes], List[float]]] = None,
        identity: Optional[str] = None,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
        enforce_detection: bool = True,
        align: bool = True,
        normalization: str = "base",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Verify identity either by direct comparison or against stored identity.

        Args:
            img1_path: Query image/embedding
            img2_path: Second image/embedding (for direct comparison, optional if identity provided)
            identity: Identity to verify against (optional)
            model_name: Recognition model
            detector_backend: Detector
            distance_metric: Distance metric
            enforce_detection: Enforce face detection
            align: Align faces
            normalization: Normalization
            threshold: Verification threshold

        Returns:
            Verification result dictionary
        """
        pass

    @abstractmethod
    def get_identity_embedding(
        self,
        identity: str,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        align: bool = True,
        normalization: str = "base",
        l2_normalize: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored embedding for an identity.

        Args:
            identity: Identity identifier
            model_name: Model name
            detector_backend: Detector used
            align: Alignment flag
            normalization: Normalization
            l2_normalize: L2 norm flag

        Returns:
            Embedding record or None
        """
        pass

    @abstractmethod
    def delete_identity(
        self,
        identity: str,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        align: bool = True,
        normalization: str = "base",
        l2_normalize: bool = False,
    ) -> bool:
        """
        Delete identity from backend.

        Args:
            identity: Identity to delete
            model_name: Model name
            detector_backend: Detector
            align: Alignment
            normalization: Normalization
            l2_normalize: L2 norm

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def list_identities(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
    ) -> List[str]:
        """
        List all registered identities.

        Args:
            model_name: Filter by model (optional)
            detector_backend: Filter by detector (optional)

        Returns:
            List of identity strings
        """
        pass

    @abstractmethod
    def count(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
    ) -> int:
        """
        Count stored embeddings.

        Args:
            model_name: Filter by model
            detector_backend: Filter by detector

        Returns:
            Count of embeddings
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close backend connections."""
        pass