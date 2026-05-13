# built-in dependencies
import os
import json
import hashlib
import struct
from typing import Any, Dict, Optional, List, Union

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.modeling import build_model
from deepface.commons.logger import Logger

logger = Logger()


class MilvusClient(Database):
    """
    Milvus client for storing and retrieving face embeddings and indices.
    """

    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        try:
            from pymilvus import MilvusClient, DataType
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "pymilvus is an optional dependency. Install with 'pip install pymilvus'"
            ) from e

        self.MilvusClient = MilvusClient
        self.DataType = DataType

        if connection is not None:
            self.client = connection
        else:
            self.conn_details = connection_details or os.environ.get("DEEPFACE_MILVUS_URI", "http://localhost:19530")
            if not isinstance(self.conn_details, str):
                raise ValueError(
                    "Milvus URI must be provided as a string in connection_details "
                    "or via DEEPFACE_MILVUS_URI environment variable."
                )

            self.client = self.MilvusClient(uri=self.conn_details)

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure Milvus collection exists.
        """
        model_name = kwargs.get("model_name", "VGG-Face")
        detector_backend = kwargs.get("detector_backend", "opencv")
        aligned = kwargs.get("aligned", True)
        l2_normalized = kwargs.get("l2_normalized", False)

        collection_name = self.__generate_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        if self.client.has_collection(collection_name):
            logger.debug(f"Milvus collection '{collection_name}' already exists.")
            return

        model = build_model(task="facial_recognition", model_name=model_name)
        dimensions = model.output_shape

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )
        schema.add_field(field_name="id", datatype=self.DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=self.DataType.FLOAT_VECTOR, dim=dimensions)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE" if l2_normalized else "L2"
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.debug(f"Created Milvus collection '{collection_name}' with dimension {dimensions}.")

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert embeddings into Milvus database in batches.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        self.initialize_database(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )

        collection_name = self.__generate_collection_name(
            embeddings[0]["model_name"],
            embeddings[0]["detector_backend"],
            embeddings[0]["aligned"],
            embeddings[0]["l2_normalized"],
        )

        total = 0
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            data = []
            for e in batch:
                face_json = json.dumps(e["face"].tolist())
                face_hash = hashlib.sha256(face_json.encode()).hexdigest()
                embedding_bytes = struct.pack(f'{len(e["embedding"])}d', *e["embedding"])
                embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

                emb_id = hash(f"{face_hash}:{embedding_hash}") % (2**63)

                data.append({
                    "id": emb_id,
                    "embedding": e["embedding"],
                    "img_name": e["img_name"],
                    # Store other metadata as dynamic fields
                })
            self.client.insert(collection_name=collection_name, data=data)
            total += len(data)

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

        collection_name = self.__generate_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        search_params = {
            "metric_type": "COSINE" if l2_normalized else "L2",
            "params": {}
        }

        results = self.client.search(
            collection_name=collection_name,
            data=[vector],
            anns_field="embedding",
            search_params=search_params,
            limit=limit,
            output_fields=["img_name"]
        )

        for hit in results[0]:
            distance = hit['distance']
            out.append({
                "id": hit['id'],
                "distance": distance,
                "img_name": hit['entity']['img_name'],
            })

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
        Fetch all embeddings from Milvus database.
        """
        out: List[Dict[str, Any]] = []

        collection_name = self.__generate_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        if not self.client.has_collection(collection_name):
            return out

        # Query all
        results = self.client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id", "embedding", "img_name"],
            limit=16384  # Milvus limit
        )

        for res in results:
            out.append({
                "id": res["id"],
                "embedding": res["embedding"],
                "img_name": res["img_name"],
            })

        return out

    def close(self) -> None:
        """Close the Milvus client"""
        self.client.close()

    @staticmethod
    def __generate_collection_name(
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> str:
        """
        Generate Milvus collection name based on parameters.
        """
        collection_name_attributes = [
            "embeddings",
            model_name.replace("-", ""),
            detector_backend,
            "Aligned" if aligned else "Unaligned",
            "Norm" if l2_normalized else "Raw",
        ]
        return "_".join(collection_name_attributes).lower()