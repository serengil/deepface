import os
import json
import hashlib
import struct
import uuid
from typing import Any, Dict, Optional, List, Union

from deepface.modules.database.types import Database
from deepface.modules.modeling import build_model


class QdrantClient(Database):
    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        try:
            from qdrant_client import QdrantClient as _QdrantClient
            from qdrant_client import models
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "qdrant-client is an optional dependency. "
                "Install with 'pip install qdrant-client'"
            ) from e

        self.models = models

        if connection is not None:
            self._client = connection
        else:
            details = connection_details or os.environ.get("DEEPFACE_QDRANT_URI")
            if not details or not isinstance(details, (str, dict)):
                raise ValueError(
                    "Qdrant connection_details must be a URL string or a non-empty dictionary, "
                    "or DEEPFACE_QDRANT_URI must contain a URL."
                )
            client_options = details if isinstance(details, dict) else {"url": details}
            self._client = _QdrantClient(**client_options)

    def initialize_database(self, **kwargs: Any) -> None:
        model_name = kwargs.get("model_name", "VGG-Face")
        detector_backend = kwargs.get("detector_backend", "opencv")
        aligned = kwargs.get("aligned", True)
        l2_normalized = kwargs.get("l2_normalized", False)

        collection_name = self.__generate_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        if self._client.collection_exists(collection_name):
            return

        dimensions = build_model(
            task="facial_recognition", model_name=model_name
        ).output_shape
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=self.models.VectorParams(
                size=dimensions,
                distance=(
                    self.models.Distance.COSINE
                    if l2_normalized
                    else self.models.Distance.EUCLID
                ),
            ),
        )

    def insert_embeddings(
        self, embeddings: List[Dict[str, Any]], batch_size: int = 100
    ) -> int:
        if not embeddings:
            raise ValueError("No embeddings to insert.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        first = embeddings[0]
        self.initialize_database(
            model_name=first["model_name"],
            detector_backend=first["detector_backend"],
            aligned=first["aligned"],
            l2_normalized=first["l2_normalized"],
        )

        collection_name = self.__generate_collection_name(
            first["model_name"],
            first["detector_backend"],
            first["aligned"],
            first["l2_normalized"],
        )

        for i in range(0, len(embeddings), batch_size):
            points = []
            for e in embeddings[i : i + batch_size]:
                face_json = json.dumps(e["face"].tolist())
                face_hash = hashlib.sha256(face_json.encode()).hexdigest()
                embedding_bytes = struct.pack(
                    f'{len(e["embedding"])}d', *e["embedding"]
                )
                embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_OID, f"{face_hash}:{embedding_hash}")
                )

                points.append(
                    self.models.PointStruct(
                        id=point_id,
                        vector=e["embedding"],
                        payload={"img_name": e["img_name"]},
                    )
                )
            self._client.upsert(
                collection_name=collection_name, points=points, wait=True
            )

        return len(embeddings)

    def search_by_vector(
        self,
        vector: List[float],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        aligned: bool = True,
        l2_normalized: bool = False,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        collection_name = self.__generate_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        results = self._client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "id": hit.id,
                "distance": (
                    max(0.0, 1.0 - float(hit.score))
                    if l2_normalized
                    else float(hit.score)
                ),
                "img_name": hit.payload.get("img_name") if hit.payload else None,
            }
            for hit in results.points
        ]

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        collection_name = self.__generate_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        if not self._client.collection_exists(collection_name):
            return out

        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            for record in records:
                out.append(
                    {
                        "id": record.id,
                        "embedding": record.vector,
                        "img_name": (record.payload or {}).get("img_name"),
                    }
                )

            if offset is None:
                break

        return out

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def __generate_collection_name(
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> str:
        alignment = "aligned" if aligned else "unaligned"
        normalization = "norm" if l2_normalized else "raw"
        return (
            f"embeddings_{model_name.replace('-', '')}_{detector_backend}_"
            f"{alignment}_{normalization}"
        ).lower()
