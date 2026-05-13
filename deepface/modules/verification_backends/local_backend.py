# built-in dependencies
from typing import Dict, Any, Optional, Union, List, cast
import time
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.modules.verification import (
    find_distance,
    find_threshold,
    find_confidence,
    __extract_faces_and_embeddings as extract_faces_and_embeddings,
)
from deepface.modules.verification_backends.base import VerificationBackend
from deepface.commons.logger import Logger

logger = Logger()


class LocalVerificationBackend(VerificationBackend):
    """
    Local verification backend using direct image comparison.
    This maintains the original DeepFace.verify() behavior.
    """

    def connect(self) -> None:
        # No connection needed for local verification
        pass

    def register(
        self,
        img_path: Union[str, np.ndarray, Any, List[float]],
        identity: str,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        align: bool = True,
        normalization: str = "base",
        l2_normalize: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        For local backend, registration just stores identity in a simple dict.
        Note: This is a placeholder; actual local registration uses datastore.
        """
        logger.warning(
            "Local verification backend does not support persistent registration. "
            "Use DeepFace.register() with database backend instead."
        )
        return {"registered": False, "identity": identity, "message": "Local backend does not store embeddings"}

    def verify(
        self,
        img1_path: Union[str, np.ndarray, Any, List[float]],
        img2_path: Optional[Union[str, np.ndarray, Any, List[float]]] = None,
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
        Direct image-to-image verification (original DeepFace.verify behavior).
        """
        if img2_path is None:
            raise ValueError("img2_path is required for local verification backend")

        tic = time.time()

        # Extract embeddings
        img1_embeddings, img1_facial_areas = extract_faces_and_embeddings(
            img_path=img1_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization,
            anti_spoofing=kwargs.get("anti_spoofing", False),
        )

        img2_embeddings, img2_facial_areas = extract_faces_and_embeddings(
            img_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization,
            anti_spoofing=kwargs.get("anti_spoofing", False),
        )

        # Find minimum distance
        min_distance, min_idx, min_idy = float("inf"), None, None
        for idx, emb1 in enumerate(img1_embeddings):
            for idy, emb2 in enumerate(img2_embeddings):
                distance = float(find_distance(emb1, emb2, distance_metric))
                if distance < min_distance:
                    min_distance, min_idx, min_idy = distance, idx, idy

        pretuned = find_threshold(model_name, distance_metric)
        threshold_val = threshold or pretuned
        confidence = find_confidence(
            distance=min_distance,
            model_name=model_name,
            distance_metric=distance_metric,
            verified=min_distance <= pretuned,
        )

        no_facial_area = {"x": None, "y": None, "w": None, "h": None, "left_eye": None, "right_eye": None}
        facial_areas = (
            no_facial_area if min_idx is None else img1_facial_areas[min_idx],
            no_facial_area if min_idy is None else img2_facial_areas[min_idy],
        )

        toc = time.time()

        return {
            "verified": min_distance <= threshold_val,
            "distance": float(min_distance),
            "threshold": threshold_val,
            "confidence": confidence,
            "model": model_name,
            "detector_backend": detector_backend,
            "similarity_metric": distance_metric,
            "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
            "time": round(toc - tic, 2),
        }

    def get_identity_embedding(
        self,
        identity: str,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        align: bool = True,
        normalization: str = "base",
        l2_normalize: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Local backend doesn't store identity embeddings."""
        return None

    def delete_identity(
        self,
        identity: str,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        align: bool = True,
        normalization: str = "base",
        l2_normalize: bool = False,
    ) -> bool:
        """Local backend does not support deletion."""
        return False

    def list_identities(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
    ) -> List[str]:
        """Local backend has no identity registry."""
        return []

    def count(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
    ) -> int:
        """Local backend count is zero."""
        return 0

    def close(self) -> None:
        pass