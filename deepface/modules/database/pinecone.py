# built-in dependencies
import os
import json
import hashlib
import struct
import math
from typing import Any, Dict, Optional, List, Union

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.modeling import build_model
from deepface.commons.logger import Logger

logger = Logger()


class PineconeClient(Database):
    """
    Pinecone client for storing and retrieving face embeddings and indices.
    """

    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "pinecone is an optional dependency. Install with 'pip install pinecone'"
            ) from e

        self.pinecone = Pinecone
        self.serverless_spec = ServerlessSpec

        if connection is not None:
            self.client = connection
        else:
            self.conn_details = connection_details or os.environ.get("DEEPFACE_PINECONE_API_KEY")
            if not isinstance(self.conn_details, str):
                raise ValueError(
                    "Pinecone api key must be provided as a string in connection_details "
                    "or via DEEPFACE_PINECONE_API_KEY environment variable."
                )

            self.client = self.pinecone(api_key=self.conn_details)

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure Pinecone index exists.
        """
        model_name = kwargs.get("model_name", "VGG-Face")
        detector_backend = kwargs.get("detector_backend", "opencv")
        aligned = kwargs.get("aligned", True)
        l2_normalized = kwargs.get("l2_normalized", False)

        index_name = self.__generate_index_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        if self.client.has_index(index_name):
            logger.debug(f"Pinecone index '{index_name}' already exists.")
            return

        model = build_model(task="facial_recognition", model_name=model_name)
        dimensions = model.output_shape
        similarity_function = "cosine" if l2_normalized else "euclidean"

        self.client.create_index(
            name=index_name,
            dimension=dimensions,
            metric=similarity_function,
            spec=self.serverless_spec(
                cloud=os.getenv("DEEPFACE_PINECONE_CLOUD", "aws"),
                region=os.getenv("DEEPFACE_PINECONE_REGION", "us-east-1"),
            ),
        )
        logger.debug(f"Created Pinecone index '{index_name}' with dimension {dimensions}.")

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert embeddings into Pinecone database in batches.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        self.initialize_database(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )

        index_name = self.__generate_index_name(
            embeddings[0]["model_name"],
            embeddings[0]["detector_backend"],
            embeddings[0]["aligned"],
            embeddings[0]["l2_normalized"],
        )

        # connect to the index
        index = self.client.Index(index_name)

        total = 0
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            vectors = []
            for e in batch:
                face_json = json.dumps(e["face"].tolist())
                face_hash = hashlib.sha256(face_json.encode()).hexdigest()
                embedding_bytes = struct.pack(f'{len(e["embedding"])}d', *e["embedding"])
                embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

                vectors.append(
                    {
                        "id": f"{face_hash}:{embedding_hash}",
                        "values": e["embedding"],
                        "metadata": {
                            "img_name": e["img_name"],
                            # "face": e["face"].tolist(),
                            # "face_shape": list(e["face"].shape),
                        },
                    }
                )
            index.upsert(vectors=vectors)
            total += len(vectors)

        return total

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
        out: List[Dict[str, Any]] = []

        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        index_name = self.__generate_index_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        index = self.client.Index(index_name)
        results = index.query(
            vector=vector,
            top_k=limit,
            include_metadata=True,
            include_values=False,
        )

        if not results.matches:
            return out

        for res in results.matches:
            score = float(res.score)
            if l2_normalized:
                distance = 1 - score
            else:
                distance = math.sqrt(max(score, 0.0))

            out.append(
                {
                    "id": res.id,
                    "distance": distance,
                    "img_name": res.metadata.get("img_name"),
                }
            )
        return out

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all embeddings from Pinecone database in batches.
        """
        out: List[Dict[str, Any]] = []

        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        index_name = self.__generate_index_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        index = self.client.Index(index_name)

        # Fetch all IDs
        ids: List[str] = []
        for _id in index.list():
            ids.extend(_id)

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            fetched = index.fetch(ids=batch_ids)
            for _id, v in fetched.get("vectors", {}).items():
                md = v.get("metadata") or {}
                out.append(
                    {
                        "id": _id,
                        "embedding": v.get("values"),
                        "img_name": md.get("img_name"),
                        "face_hash": md.get("face_hash"),
                        "embedding_hash": md.get("embedding_hash"),
                    }
                )

        return out

    def close(self) -> None:
        """Pinecone client does not require explicit closure"""
        return

    def search_by_identity(
        self,
        identity: str,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        align: bool = True,
        l2_normalize: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        index_name = self.__generate_index_name(model_name, detector_backend, align, l2_normalize)
        self.initialize_database(model_name=model_name, detector_backend=detector_backend, aligned=align, l2_normalized=l2_normalize)
        index = self.client.Index(index_name)
        dim = self.__get_dimension(model_name)
        dummy_vector = [0.0] * dim
        filter_expr = {"img_name": {"$eq": identity}}
        query_limit = limit if limit is not None else 100
        results = index.query(vector=dummy_vector, top_k=query_limit, filter=filter_expr, include_metadata=True, include_values=True)
        out = []
        for match in results.matches:
            meta = match.metadata or {}
            out.append({
                "id": match.id,
                "embedding": match.values,
                "metadata": {
                    "img_name": meta.get("img_name"),
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                    "aligned": align,
                    "l2_normalized": l2_normalize,
                },
            })
        return out

    def count(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        align: Optional[bool] = None,
        l2_normalize: Optional[bool] = None,
    ) -> int:
        total = 0
        if (
            model_name is not None
            and detector_backend is not None
            and align is not None
            and l2_normalize is not None
        ):
            index_name = self.__generate_index_name(model_name, detector_backend, align, l2_normalize)
            self.initialize_database(model_name=model_name, detector_backend=detector_backend, aligned=align, l2_normalized=l2_normalize)
            index = self.client.Index(index_name)
            stats = index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
        else:
            for index_name in self.client.list_indexes().names:
                index = self.client.Index(index_name)
                stats = index.describe_index_stats()
                total += stats.get("total_vector_count", 0)
        return total

    def delete(
        self,
        ids: List[int],
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        align: bool = True,
        l2_normalize: bool = False,
    ) -> bool:
        if not ids:
            return False
        if model_name is None or detector_backend is None:
            return False
        index_name = self.__generate_index_name(model_name, detector_backend, align, l2_normalize)
        self.initialize_database(model_name=model_name, detector_backend=detector_backend, aligned=align, l2_normalized=l2_normalize)
        index = self.client.Index(index_name)
        str_ids = [str(i) for i in ids]
        index.delete(ids=str_ids)
        return True

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
        if not model_name or not detector_backend:
            return False
        index_name = self.__generate_index_name(model_name, detector_backend, align, l2_normalize)
        self.initialize_database(model_name=model_name, detector_backend=detector_backend, aligned=align, l2_normalized=l2_normalize)
        index = self.client.Index(index_name)
        vector_id = str(id)
        upsert_vector = {
            "id": vector_id,
            "values": embedding if embedding is not None else [],
            "metadata": metadata or {},
        }
        index.upsert(vectors=[upsert_vector])
        return True

    @staticmethod
    def __get_dimension(model_name: str) -> int:
        """Get embedding dimension for a given model."""
        model = build_model(task="facial_recognition", model_name=model_name)
        return model.output_shape

    @staticmethod
    def __generate_index_name(
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> str:
        """
        Generate Pinecone index name based on parameters.
        """
        index_name_attributes = [
            "embeddings",
            model_name.replace("-", ""),
            detector_backend,
            "Aligned" if aligned else "Unaligned",
            "Norm" if l2_normalized else "Raw",
        ]
        return "-".join(index_name_attributes).lower()
