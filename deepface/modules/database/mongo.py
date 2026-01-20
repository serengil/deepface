# built-in dependencies
import os
import json
import hashlib
import struct
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.exceptions import DuplicateEntryError
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=too-many-positional-arguments, too-many-instance-attributes
class MongoDbClient(Database):
    """
    MongoDB equivalent of PostgresClient for DeepFace embeddings storage.
    """

    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
        db_name: str = "deepface",
    ) -> None:
        try:
            from pymongo import MongoClient, ASCENDING
            from pymongo.errors import DuplicateKeyError, BulkWriteError
            from bson import Binary
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "pymongo is an optional dependency. Please install it as `pip install pymongo`"
            ) from e
        self.MongoClient = MongoClient
        self.ASCENDING = ASCENDING
        self.DuplicateKeyError = DuplicateKeyError
        self.BulkWriteError = BulkWriteError
        self.Binary = Binary

        if connection is not None:
            self.client = connection
        else:
            self.conn_details = connection_details or os.environ.get("DEEPFACE_MONGO_URI")
            if not self.conn_details:
                raise ValueError(
                    "MongoDB connection information not found. "
                    "Please provide connection_details or set DEEPFACE_MONGO_URI"
                )

            if isinstance(self.conn_details, str):
                self.client = MongoClient(self.conn_details)
            else:
                self.client = MongoClient(**self.conn_details)

        self.db = self.client[db_name]
        self.embeddings = self.db.embeddings
        self.embeddings_index = self.db.embeddings_index
        self.counters = self.db.counters
        self.initialize_database()

    def close(self) -> None:
        """Close MongoDB connection."""
        self.client.close()

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure required MongoDB indexes exist.
        """

        # Unique constraint for embeddings
        self.embeddings.create_index(
            [("face_hash", self.ASCENDING), ("embedding_hash", self.ASCENDING)],
            unique=True,
            name="uniq_face_embedding",
        )

        # Unique constraint for embeddings_index
        self.embeddings_index.create_index(
            [
                ("model_name", self.ASCENDING),
                ("detector_backend", self.ASCENDING),
                ("align", self.ASCENDING),
                ("l2_normalized", self.ASCENDING),
            ],
            unique=True,
            name="uniq_index_config",
        )

        # counters collection for auto-incrementing IDs
        if not self.counters.find_one({"_id": "embedding_id"}):
            self.counters.insert_one({"_id": "embedding_id", "seq": 0})

        logger.debug("MongoDB indexes ensured.")

    def upsert_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        index_data: bytes,
    ) -> None:
        """
        Upsert embeddings index into MongoDB.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            index_data (bytes): Serialized index data.
        """
        self.embeddings_index.update_one(
            {
                "model_name": model_name,
                "detector_backend": detector_backend,
                "align": aligned,
                "l2_normalized": l2_normalized,
            },
            {
                "$set": {
                    "index_data": self.Binary(index_data),
                    "updated_at": datetime.now(timezone.utc),
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc),
                },
            },
            upsert=True,
        )

    def get_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> bytes:
        """
        Retrieve embeddings index from MongoDB.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
        Returns:
            bytes: Serialized index data.
        """
        doc = self.embeddings_index.find_one(
            {
                "model_name": model_name,
                "detector_backend": detector_backend,
                "align": aligned,
                "l2_normalized": l2_normalized,
            },
            {"index_data": 1},
        )

        if not doc:
            raise ValueError(
                "No Embeddings index found for the specified parameters "
                f"{model_name=}, {detector_backend=}, "
                f"{aligned=}, {l2_normalized=}. "
                "You must run build_index first."
            )

        return bytes(doc["index_data"])

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert embeddings into MongoDB.
        Args:
            embeddings (List[Dict[str, Any]]): List of embedding records to insert.
            batch_size (int): Number of records to insert in each batch.
        Returns:
            int: Number of embeddings successfully inserted.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        docs: List[Dict[str, Any]] = []

        for e in embeddings:
            face = e["face"]
            face_shape = list(face.shape)

            binary_face_data = self.Binary(face.astype(np.float32).tobytes())

            embedding_bytes = struct.pack(f'{len(e["embedding"])}d', *e["embedding"])

            face_hash = hashlib.sha256(json.dumps(face.tolist()).encode()).hexdigest()
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

            int_id = self.counters.find_one_and_update(
                {"_id": "embedding_id"}, {"$inc": {"seq": 1}}, upsert=True, return_document=True
            )["seq"]

            docs.append(
                {
                    "sequence": int_id,
                    "img_name": e["img_name"],
                    "face": binary_face_data,
                    "face_shape": face_shape,
                    "model_name": e["model_name"],
                    "detector_backend": e["detector_backend"],
                    "aligned": e["aligned"],
                    "l2_normalized": e["l2_normalized"],
                    "embedding": e["embedding"],
                    "face_hash": face_hash,
                    "embedding_hash": embedding_hash,
                    "created_at": datetime.now(timezone.utc),
                }
            )

        inserted = 0
        try:
            for i in range(0, len(docs), batch_size):
                result = self.embeddings.insert_many(docs[i : i + batch_size], ordered=False)
                inserted += len(result.inserted_ids)
        except (self.DuplicateKeyError, self.BulkWriteError) as e:
            if len(docs) == 1:
                logger.warn("Duplicate detected for extracted face and embedding.")
                return inserted
            raise DuplicateEntryError(
                f"Duplicate detected for extracted face and embedding in {i}-th batch"
            ) from e

        return inserted

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all embeddings from MongoDB based on specified parameters.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            batch_size (int): Number of records to fetch in each batch.
        Returns:
            List[Dict[str, Any]]: List of embedding records.
        """

        cursor = self.embeddings.find(
            {
                "model_name": model_name,
                "detector_backend": detector_backend,
                "aligned": aligned,
                "l2_normalized": l2_normalized,
            },
            {
                "_id": 1,
                "sequence": 1,
                "img_name": 1,
                "embedding": 1,
            },
            batch_size=batch_size,
        ).sort("sequence", self.ASCENDING)

        results: List[Dict[str, Any]] = []
        for doc in cursor:
            results.append(
                {
                    "_id": str(doc["_id"]),
                    "id": doc["sequence"],
                    "img_name": doc["img_name"],
                    "embedding": doc["embedding"],
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                    "aligned": aligned,
                    "l2_normalized": l2_normalized,
                }
            )

        return results

    def search_by_id(
        self,
        ids: Union[List[str], List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Search records by their IDs.
        """
        cursor = self.embeddings.find(
            {"sequence": {"$in": ids}},
            {
                "_id": 1,
                "sequence": 1,
                "img_name": 1,
            },
        )

        results: List[Dict[str, Any]] = []
        for doc in cursor:
            results.append(
                {
                    "_id": str(doc["_id"]),
                    "id": doc["sequence"],
                    "img_name": doc["img_name"],
                }
            )

        return results
