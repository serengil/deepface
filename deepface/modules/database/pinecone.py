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
