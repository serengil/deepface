# built-in dependencies
import os
import json
import hashlib
import struct
import base64
import uuid
import math
from typing import Any, Dict, Optional, List, Union

# project dependencies
from deepface.modules.database.types import Database
from deepface.commons.logger import Logger

logger = Logger()


_SCHEMA_CHECKED: Dict[str, bool] = {}


# pylint: disable=too-many-positional-arguments
class WeaviateClient(Database):
    """
    Weaviate client for storing and retrieving face embeddings and indices.
    """

    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        try:
            import weaviate
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "weaviate-client is an optional dependency. "
                "Install with 'pip install weaviate-client'"
            ) from e

        self.weaviate = weaviate

        if connection is not None:
            self.client = connection
            # URL key for _WEAVIATE_CHECKED; fallback if client has no URL
            self.url = getattr(connection, "url", str(id(connection)))
        else:
            self.conn_details = connection_details or os.environ.get("DEEPFACE_WEAVIATE_URL")
            if isinstance(self.conn_details, str):
                self.url = self.conn_details
                self.api_key = os.getenv("WEAVIATE_API_KEY")
            elif isinstance(self.conn_details, dict):
                self.url = self.conn_details.get("url")
                self.api_key = self.conn_details.get("api_key") or os.getenv("WEAVIATE_API_KEY")
            else:
                raise ValueError("connection_details must be a string or dict with 'url'.")

            if not self.url:
                raise ValueError("Weaviate URL not provided in connection_details.")

            client_config = {"url": self.url}
            if getattr(self, "api_key", None):
                client_config["auth_client_secret"] = self.weaviate.AuthApiKey(api_key=self.api_key)

            self.client = self.weaviate.Client(**client_config)

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure Weaviate schemas exist for embeddings using both cosine and L2 (euclidean).
        """
        model_name = kwargs.get("model_name", "VGG-Face")
        detector_backend = kwargs.get("detector_backend", "opencv")
        aligned = kwargs.get("aligned", True)
        l2_normalized = kwargs.get("l2_normalized", False)

        existing_schema = self.client.schema.get()
        existing_classes = {c["class"] for c in existing_schema.get("classes", [])}

        class_name = self.__generate_class_name(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        if _SCHEMA_CHECKED.get(class_name):
            logger.debug("Weaviate schema already checked, skipping.")
            return

        if class_name in existing_classes:
            logger.debug(f"Weaviate class {class_name} already exists.")
            return

        self.client.schema.create_class(
            {
                "class": class_name,
                "vectorIndexType": "hnsw",
                "vectorizer": "none",
                "vectorIndexConfig": {
                    "M": int(os.getenv("WEAVIATE_HNSW_M", "16")),
                    "distance": "cosine" if l2_normalized else "l2-squared",
                },
                "properties": [
                    {"name": "img_name", "dataType": ["text"]},
                    {"name": "face", "dataType": ["blob"]},
                    {"name": "face_shape", "dataType": ["int[]"]},
                    {"name": "model_name", "dataType": ["text"]},
                    {"name": "detector_backend", "dataType": ["text"]},
                    {"name": "aligned", "dataType": ["boolean"]},
                    {"name": "l2_normalized", "dataType": ["boolean"]},
                    {"name": "face_hash", "dataType": ["text"]},
                    {"name": "embedding_hash", "dataType": ["text"]},
                    # embedding property is optional since we pass it as vector
                    {"name": "embedding", "dataType": ["number[]"]},
                ],
            }
        )

        logger.debug(f"Weaviate class {class_name} created successfully.")
        _SCHEMA_CHECKED[class_name] = True

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert multiple embeddings into Weaviate using batch API.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        self.initialize_database(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )
        class_name = self.__generate_class_name(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )

        with self.client.batch as batcher:
            batcher.batch_size = batch_size
            batcher.timeout_retries = 3
            for e in embeddings:
                face_json = json.dumps(e["face"].tolist())
                face_hash = hashlib.sha256(face_json.encode()).hexdigest()
                embedding_bytes = struct.pack(f'{len(e["embedding"])}d', *e["embedding"])
                embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

                # Check if embedding already exists
                query = (
                    self.client.query.get(class_name, ["embedding_hash"])
                    .with_where(
                        {
                            "path": ["embedding_hash"],
                            "operator": "Equal",
                            "valueText": embedding_hash,
                        }
                    )
                    .with_limit(1)
                    .do()
                )
                existing = query.get("data", {}).get("Get", {}).get(class_name, [])
                if existing:
                    logger.warn(
                        f"Embedding with hash {embedding_hash} already exists in {class_name}."
                    )
                    continue

                uid = str(uuid.uuid4())
                properties = {
                    "img_name": e["img_name"],
                    "face": base64.b64encode(e["face"].tobytes()).decode("utf-8"),
                    "face_shape": list(e["face"].shape),
                    "model_name": e["model_name"],
                    "detector_backend": e["detector_backend"],
                    "aligned": e["aligned"],
                    "l2_normalized": e["l2_normalized"],
                    "embedding": e["embedding"],  # optional
                    "face_hash": face_hash,
                    "embedding_hash": embedding_hash,
                }

                batcher.add_data_object(properties, class_name, vector=e["embedding"], uuid=uid)

        return len(embeddings)

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all embeddings with filters.
        """
        class_name = self.__generate_class_name(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )
        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        results = (
            self.client.query.get(class_name, ["img_name", "embedding"])
            .with_additional(["id"])
            .do()
        )
        data = results.get("data", {}).get("Get", {}).get(class_name, [])

        embeddings = []
        for r in data:
            embeddings.append(
                {
                    "id": r.get("_additional", {}).get("id"),
                    "img_name": r["img_name"],
                    "embedding": r["embedding"],
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                    "aligned": aligned,
                    "l2_normalized": l2_normalized,
                }
            )
        return embeddings

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
        class_name = self.__generate_class_name(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )
        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        query = self.client.query.get(class_name, ["img_name", "embedding"])
        query = (
            query.with_near_vector({"vector": vector})
            .with_limit(limit)
            .with_additional(["id", "distance"])
        )
        results = query.do()

        data = results.get("data", {}).get("Get", {}).get(class_name, [])

        return [
            {
                "id": r.get("_additional", {}).get("id"),
                "img_name": r["img_name"],
                "embedding": r["embedding"],
                "distance": (
                    r.get("_additional", {}).get("distance")
                    if l2_normalized
                    else math.sqrt(r.get("_additional", {}).get("distance"))
                ),
            }
            for r in data
        ]

    def close(self) -> None:
        """
        Close the Weaviate client connection.
        """
        self.client.close()

    @staticmethod
    def __generate_class_name(
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> str:
        """
        Generate Weaviate class name based on parameters.
        """
        class_name_attributes = [
            model_name.replace("-", ""),
            detector_backend,
            "Aligned" if aligned else "Unaligned",
            "Norm" if l2_normalized else "Raw",
        ]
        return "Embeddings_" + "_".join(class_name_attributes).lower()
