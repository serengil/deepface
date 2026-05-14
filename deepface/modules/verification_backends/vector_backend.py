# built-in dependencies
from typing import Dict, Any, Optional, Union, List
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
from deepface.modules.database.inventory import database_inventory
from deepface.modules.database.types import Database
from deepface.commons.logger import Logger

logger = Logger()


class VectorVerificationBackend(VerificationBackend):
    """
    Vector database-based verification backend.
    Uses Database clients for identity lookup and verification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.db_client: Optional[Database] = None
        self.store_type = self.config.get('store_type', 'milvus')
        self.store_config = self.config.get('store_config', {})

    def connect(self) -> None:
        if self.store_type not in database_inventory:
            raise ValueError(f"Unsupported verification backend store type: {self.store_type}")
        client_class = database_inventory[self.store_type]["client"]
        self.db_client = client_class(
            connection_details=self.store_config,
            connection=self.config.get('connection')
        )

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
        Register an identity by generating embedding and storing in database.
        """
        from deepface.modules import representation

        if isinstance(img_path, list):
            embedding = img_path
            face = None
        else:
            results = representation.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=kwargs.get("enforce_detection", True),
                align=align,
                normalization=normalization,
                anti_spoofing=kwargs.get("anti_spoofing", False),
                l2_normalize=l2_normalize,
                return_face=True,
            )
            if not results:
                raise ValueError("No face detected in the image")
            result = results[0] if isinstance(results, list) else results
            embedding = result["embedding"]
            face = result.get("face")

        # Build embedding record
        record: Dict[str, Any] = {
            "id": None,
            "img_name": identity,
            "face": face,
            "model_name": model_name,
            "detector_backend": detector_backend,
            "embedding": embedding,
            "aligned": align,
            "l2_normalized": l2_normalize,
        }

        inserted = self.db_client.insert_embeddings([record])
        logger.debug(f"Inserted {inserted} embedding(s)")

        # Attempt to retrieve assigned ID after insertion
        embedding_id = None
        try:
            id_results = self.db_client.search_by_identity(
                identity=identity,
                model_name=model_name,
                detector_backend=detector_backend,
                align=align,
                l2_normalize=l2_normalize,
                limit=1,
            )
            if id_results:
                embedding_id = id_results[0]["id"]
        except NotImplementedError:
            pass

        return {
            "registered": True,
            "identity": identity,
            "embedding_id": embedding_id,
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
        l2_normalize = kwargs.get("l2_normalize", False)
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
                l2_normalize=l2_normalize,
            )
            if not img_objs:
                raise ValueError("No face detected in query image")
            query_embedding = img_objs[0][0]

        if identity is not None:
            record = self.get_identity_embedding(
                identity=identity,
                model_name=model_name,
                detector_backend=detector_backend,
                align=align,
                normalization=normalization,
                l2_normalize=l2_normalize,
            )
            if record is None:
                raise ValueError(f"Identity '{identity}' not found in database")
            stored_embedding = record["embedding"]
        elif img2_path is not None:
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
                    l2_normalize=l2_normalize,
                )
                if not img_objs2:
                    raise ValueError("No face detected in second image")
                stored_embedding = img_objs2[0][0]

                if len(query_embedding) != len(stored_embedding):
                    raise ValueError(
                        f"Embedding dimension mismatch: query has {len(query_embedding)} dims, "
                        f"second image has {len(stored_embedding)} dims."
                    )
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
        results = self.db_client.search_by_identity(
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
        results = self.db_client.search_by_identity(
            identity=identity,
            model_name=model_name,
            detector_backend=detector_backend,
            align=align,
            l2_normalize=l2_normalize,
            limit=None,
        )
        if not results:
            return False
        ids = [r["id"] for r in results]
        return self.db_client.delete(
            ids=ids,
            model_name=model_name,
            detector_backend=detector_backend,
            align=align,
            l2_normalize=l2_normalize,
        )

    def list_identities(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
    ) -> List[str]:
        """
        List all unique identities (img_name) for given model/detector.
        """
        # Since Database does not have a generic distinct method, we implement a fallback by scanning.
        if model_name and detector_backend:
            identities = set()
            for align in [True, False]:
                for l2 in [True, False]:
                    try:
                        records = self.db_client.fetch_all_embeddings(
                            model_name=model_name,
                            detector_backend=detector_backend,
                            aligned=align,
                            l2_normalized=l2,
                        )
                        for rec in records:
                            img_name = rec.get("img_name")
                            if img_name:
                                identities.add(img_name)
                    except Exception:
                        continue
            return list(identities)
        return []

    def count(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
    ) -> int:
        total = 0
        if model_name and detector_backend:
            for align in [True, False]:
                for l2 in [True, False]:
                    try:
                        total += self.db_client.count(model_name, detector_backend, align, l2)
                    except Exception:
                        continue
        return total

    def close(self) -> None:
        if self.db_client:
            self.db_client.close()
