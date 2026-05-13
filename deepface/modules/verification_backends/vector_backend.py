# built-in dependencies
from typing import Dict, Any, Optional, Union, List, cast
import time

# 3rd party dependencies
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
from deepface.modules.vectorstores.factory import create_vector_store
from deepface.commons.logger import Logger

logger = Logger()


class VectorVerificationBackend(VerificationBackend):
    """
    Vector database-based verification backend.
    Uses vector stores for identity lookup and verification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_store = None
        self.store_type = self.config.get('store_type', 'milvus')
        self.store_config = self.config.get('store_config', {})

    def connect(self) -> None:
        self.vector_store = create_vector_store(self.store_type, self.store_config)
        self.vector_store.connect()

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
        Register an identity by generating embedding and storing in vector DB.
        """
        # Generate embedding
        if isinstance(img_path, list):
            # Pre-calculated embedding
            embedding = img_path
            facial_area = None
        else:
            # Extract face and generate embedding
            img_objs = extract_faces_and_embeddings(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=kwargs.get("enforce_detection", True),
                align=align,
                normalization=normalization,
                anti_spoofing=kwargs.get("anti_spoofing", False),
            )
            if not img_objs:
                raise ValueError("No face detected in the image")
            embedding = img_objs[0][0]  # (embeddings, facial_areas)
            facial_area = img_objs[1][0]

        # Store in vector DB
        metadata = {
            "identity": identity,
            "img_name": str(img_path) if not isinstance(img_path, list) else "precalculated",
            "model_name": model_name,
            "detector_backend": detector_backend,
            "aligned": align,
            "l2_normalized": l2_normalize,
            "normalization": normalization,
            "facial_area": facial_area,
        }

        ids = self.vector_store.register(
            embeddings=[embedding],
            metadata=[metadata],
            model_name=model_name,
            detector_backend=detector_backend,
            align=align,
            l2_normalize=l2_normalize,
        )

        return {
            "registered": True,
            "identity": identity,
            "embedding_id": ids[0] if ids else None,
            "model": model_name,
        }

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
        Verify identity using vector database.
        Either img2_path (direct comparison) or identity (stored comparison) must be provided.
        """
        tic = time.time()

        # Generate query embedding
        if isinstance(img1_path, list):
            query_embedding = img1_path
        else:
            img_objs = extract_faces_and_embeddings(
                img_path=img1_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                normalization=normalization,
                anti_spoofing=kwargs.get("anti_spoofing", False),
            )
            if not img_objs:
                raise ValueError("No face detected in query image")
            query_embedding = img_objs[0][0]

        # Handle direct image comparison (from vector backend)
        if identity is None and img2_path is not None:
            if isinstance(img2_path, list):
                stored_embedding = img2_path
            else:
                img_objs2 = extract_faces_and_embeddings(
                    img_path=img2_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                    align=align,
                    normalization=normalization,
                    anti_spoofing=kwargs.get("anti_spoofing", False),
                )
                if not img_objs2:
                    raise ValueError("No face detected in second image")
                stored_embedding = img_objs2[0][0]
        elif identity is not None:
            # Fetch identity embedding from vector store
            record = self.get_identity_embedding(
                identity=identity,
                model_name=model_name,
                detector_backend=detector_backend,
                align=align,
                normalization=normalization,
                l2_normalize=kwargs.get("l2_normalize", False),
            )
            if record is None:
                raise ValueError(f"Identity '{identity}' not found in database")
            stored_embedding = record["embedding"]
        else:
            raise ValueError("Either img2_path or identity must be provided")

        distance = float(find_distance(query_embedding, stored_embedding, distance_metric))
        pretuned = find_threshold(model_name, distance_metric)
        threshold_val = threshold or pretuned
        confidence = find_confidence(
            distance=distance,
            model_name=model_name,
            distance_metric=distance_metric,
            verified=distance <= pretuned,
        )

        toc = time.time()

        return {
            "verified": distance <= threshold_val,
            "distance": distance,
            "threshold": threshold_val,
            "confidence": confidence,
            "model": model_name,
            "detector_backend": detector_backend,
            "similarity_metric": distance_metric,
            "identity": identity,
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
        """
        Retrieve stored embedding(s) for an identity from vector store.
        """
        results = self.vector_store.search_by_identity(
            identity=identity,
            model_name=model_name,
            detector_backend=detector_backend,
            align=align,
            l2_normalize=l2_normalize,
            limit=1,
        )

        if not results:
            return None

        best = results[0]
        return {
            "id": best["id"],
            "embedding": best.get("embedding"),
            "identity": identity,
            "metadata": best["metadata"],
        }

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
        Delete all embeddings for an identity.
        Note: Requires vector store to support metadata-based deletion.
        """
        # For now, we cannot easily delete by identity filter without ID
        # Future enhancement: vector stores that support metadata filter delete
        logger.warning("Delete by identity requires vector store metadata filtering support")
        return False

    def list_identities(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
    ) -> List[str]:
        """
        List all unique identities.
        Requires vector store scan; not efficient for large DBs.
        """
        # This is a placeholder - efficient implementation depends on vector store
        identities = set()
        # Cannot easily enumerate all without full scan
        return list(identities)

    def count(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
    ) -> int:
        """
        Count total embeddings.
        """
        if model_name and detector_backend:
            return self.vector_store.count(model_name, detector_backend, True, False)  # rough estimate
        return 0

    def close(self) -> None:
        if self.vector_store:
            self.vector_store.close()