# built-in dependencies
import os
import json
import hashlib
import struct
from typing import Any, Dict, Optional, List, Union

# 3rd party dependencies
import numpy as np


# project dependencies
from deepface.modules.database.types import EmbeddingRecord
from deepface.commons.logger import Logger

logger = Logger()


class PostgresClient:
    def __init__(
        self,
        connection_details: Optional[Union[Dict[str, Any], str]] = None,
        connection: Any = None,
    ) -> None:
        # Import here to avoid mandatory dependency
        try:
            import psycopg
        except ModuleNotFoundError as e:
            raise ImportError(
                "psycopg is an optional dependency, ensure the library is installed."
                "Please install using 'pip show \"psycopg[binary]\"' "
            ) from e

        self.psycopg = psycopg

        if connection is not None:
            self.conn = connection
            return

        # Retrieve connection details from parameter or environment variable
        self.conn_details = connection_details or os.environ.get("DEEPFACE_POSTGRES_URI")
        if not self.conn_details:
            raise ValueError(
                "PostgreSQL connection information not found. "
                "Please provide connection_details or set the DEEPFACE_POSTGRES_URI"
                " environment variable."
            )

        if isinstance(self.conn_details, str):
            self.conn = self.psycopg.connect(self.conn_details)
        elif isinstance(self.conn_details, dict):
            self.conn = self.psycopg.connect(**self.conn_details)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def insert_embeddings(
        self, embeddings: List[EmbeddingRecord], batch_size: int = 100
    ) -> Optional[int]:
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        query = """
            INSERT INTO embeddings (
                img_name,
                face,
                model_name,
                detector_backend,
                aligned,
                l2_normalized,
                embedding,
                face_hash,
                embedding_hash
            )
            VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
        """

        values = []
        for e in embeddings:
            face = e["face"]
            if isinstance(face, np.ndarray):
                face_json = json.dumps(face.tolist())
            elif isinstance(face, list):
                face_json = json.dumps(face)
            else:
                raise ValueError("Face data must be a numpy array or a list.")

            embedding_bytes = struct.pack(f'{len(e["embedding"])}d', *e["embedding"])

            # uniqueness is guaranteed by face hash and embedding hash
            face_hash = hashlib.sha256(face_json.encode()).digest()
            embedding_hash = hashlib.sha256(embedding_bytes).digest()

            values.append(
                (
                    e["img_name"],
                    face_json,
                    e["model_name"],
                    e["detector_backend"],
                    e["aligned"],
                    e["l2_normalized"],
                    e["embedding"],
                    face_hash,
                    embedding_hash,
                )
            )

        try:
            with self.conn.cursor() as cur:
                for i in range(0, len(values), batch_size):
                    cur.executemany(query, values[i : i + batch_size])
                    # commit for every batch
                    self.conn.commit()
                return len(values)
        except self.psycopg.errors.UniqueViolation as e:
            raise ValueError(
                f"Duplicate detected for extracted face and embedding columns in {i}-th batch"
            ) from e
