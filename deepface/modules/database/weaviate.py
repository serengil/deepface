# built-in dependencies
import os
import json
import hashlib
import struct
import base64
import uuid
from typing import Any, Dict, Optional, List, Union

# project dependencies
from deepface.modules.database.types import Database

_WEAVIATE_CHECKED: Dict[str, bool] = {}


# pylint: disable=too-many-positional-arguments
class WeaviateClient(Database):
    """
    Weaviate client for storing and retrieving face embeddings and indices.
    Similar interface to PostgresClient.
    """

    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        try:
            import weaviate
        except ModuleNotFoundError as e:
            raise ImportError(
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

        # Schema check once per URL/connection
        if not _WEAVIATE_CHECKED.get(self.url):
            self.ensure_embeddings_table()
            _WEAVIATE_CHECKED[self.url] = True

    def ensure_embeddings_table(self) -> None:
        """
        Ensure Weaviate schemas exist for embeddings using both cosine and L2 (euclidean).
        """
        schemas = ["cosine", "l2-squared"]

        existing_schema = self.client.schema.get()
        existing_classes = {c["class"] for c in existing_schema.get("classes", [])}

        base_properties = [
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
        ]

        for schema in schemas:
            class_name = "EmbeddingsNorm" if schema == "cosine" else "EmbeddingsRaw"

            if class_name in existing_classes:
                continue

            self.client.schema.create_class(
                {
                    "class": "EmbeddingsNorm" if schema == "cosine" else "EmbeddingsRaw",
                    "vectorIndexType": "hnsw",
                    "vectorizer": "none",
                    "vectorIndexConfig": {
                        "M": int(os.getenv("WEAVIATE_HNSW_M", "16")),
                        "distance": schema,
                    },
                    "properties": base_properties,
                }
            )

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert multiple embeddings into Weaviate using batch API.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        with self.client.batch as batcher:
            batcher.batch_size = batch_size
            batcher.timeout_retries = 3

            for e in embeddings:
                class_name = "EmbeddingsNorm" if e["l2_normalized"] else "EmbeddingsRaw"
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
                    raise ValueError(
                        f"Embedding with hash {embedding_hash} already exists in the database."
                    )

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
        class_name = "EmbeddingsNorm" if l2_normalized else "EmbeddingsRaw"
        filters = {
            "operator": "And",
            "operands": [
                {"path": ["model_name"], "operator": "Equal", "valueText": model_name},
                {"path": ["detector_backend"], "operator": "Equal", "valueText": detector_backend},
                {"path": ["aligned"], "operator": "Equal", "valueBoolean": aligned},
                {"path": ["l2_normalized"], "operator": "Equal", "valueBoolean": l2_normalized},
            ],
        }

        results = (
            self.client.query.get(class_name, ["img_name", "embedding"])
            .with_where(filters)
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
        class_name = "EmbeddingsNorm" if l2_normalized else "EmbeddingsRaw"

        where_filters = [
            {"path": ["model_name"], "operator": "Equal", "valueText": model_name},
            {"path": ["detector_backend"], "operator": "Equal", "valueText": detector_backend},
            {"path": ["aligned"], "operator": "Equal", "valueBoolean": aligned},
            {"path": ["l2_normalized"], "operator": "Equal", "valueBoolean": l2_normalized},
        ]

        query = self.client.query.get(class_name, ["img_name", "embedding"])
        query = query.with_where({"operator": "And", "operands": where_filters})
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
                "distance": r.get("_additional", {}).get("distance"),
            }
            for r in data
        ]

    def close(self) -> None:
        """
        Close the Weaviate client connection.
        """
        self.client.close()

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
        raise NotImplementedError("Weaviate does not support storing embeddings index.")

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
        raise NotImplementedError("Weaviate does not support storing embeddings index.")
