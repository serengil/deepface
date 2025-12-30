# built-in dependencies
import os
import json
import hashlib
import struct
from typing import Any, Dict, Optional, List, Union, cast

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules.database.types import Database
from deepface.commons.logger import Logger

logger = Logger()


_SCHEMA_CHECKED: Dict[str, bool] = {}

CREATE_EMBEDDINGS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id BIGSERIAL PRIMARY KEY,
        img_name TEXT NOT NULL,
        face BYTEA NOT NULL,
        face_shape INT[] NOT NULL,
        model_name TEXT NOT NULL,
        detector_backend TEXT NOT NULL,
        aligned BOOLEAN DEFAULT true,
        l2_normalized BOOLEAN DEFAULT false,
        embedding FLOAT8[] NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now(),
        face_hash TEXT NOT NULL,
        embedding_hash TEXT NOT NULL,
        UNIQUE (face_hash, embedding_hash)
    );
"""

CREATE_EMBEDDINGS_INDEX_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS embeddings_index (
        id SERIAL PRIMARY KEY,
        model_name TEXT,
        detector_backend TEXT,
        align BOOL,
        l2_normalized BOOL,
        index_data BYTEA,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (model_name, detector_backend, align, l2_normalized)
    );
"""


# pylint: disable=too-many-positional-arguments
class PostgresClient(Database):
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
        else:
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

        # Ensure the embeddings table exists
        self.ensure_embeddings_table()

    def ensure_embeddings_table(self) -> None:
        """
        Ensure that the `embeddings` table exists.
        """
        dsn = self.conn.info.dsn

        if _SCHEMA_CHECKED.get(dsn):
            logger.debug("PostgreSQL schema already checked, skipping.")
            return

        with self.conn.cursor() as cur:
            try:
                cur.execute(CREATE_EMBEDDINGS_TABLE_SQL)
                logger.debug("Ensured 'embeddings' table either exists or was created in Postgres.")

                cur.execute(CREATE_EMBEDDINGS_INDEX_TABLE_SQL)
                logger.debug(
                    "Ensured 'embeddings_index' table either exists or was created in Postgres."
                )
            except Exception as e:
                if getattr(e, "sqlstate", None) == "42501":  # permission denied
                    raise ValueError(
                        "The PostgreSQL user does not have permission to create "
                        "the required tables ('embeddings', 'embeddings_index'). "
                        "Please ask your database administrator to grant CREATE privileges "
                        "on the schema."
                    ) from e
                raise

        self.conn.commit()
        _SCHEMA_CHECKED[dsn] = True

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def upsert_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        index_data: bytes,
    ) -> None:
        """
        Upsert embeddings index into PostgreSQL.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            index_data (bytes): Serialized index data.
        """
        query = """
            INSERT INTO embeddings_index (model_name, detector_backend, align, l2_normalized, index_data)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (model_name, detector_backend, align, l2_normalized)
            DO UPDATE SET
                index_data = EXCLUDED.index_data,
                updated_at = NOW()
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (model_name, detector_backend, aligned, l2_normalized, index_data),
            )
            self.conn.commit()

    def get_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> bytes:
        """
        Get embeddings index from PostgreSQL.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
        Returns:
            bytes: Serialized index data.
        """
        query = """
            SELECT index_data
            FROM embeddings_index
            WHERE model_name = %s AND detector_backend = %s AND align = %s AND l2_normalized = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (model_name, detector_backend, aligned, l2_normalized),
            )
            result = cur.fetchone()
            if result:
                return cast(bytes, result[0])
            raise ValueError(
                "No Embeddings index found for the specified parameters "
                f" {model_name=}, {detector_backend=}, {aligned=}, {l2_normalized=}. "
                "You must run build_index first."
            )

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert multiple embeddings into PostgreSQL.
        Args:
            embeddings (List[Dict[str, Any]]): List of embeddings to insert.
            batch_size (int): Number of embeddings to insert per batch.
        Returns:
            int: Number of embeddings inserted.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        query = """
            INSERT INTO embeddings (
                img_name,
                face,
                face_shape,
                model_name,
                detector_backend,
                aligned,
                l2_normalized,
                embedding,
                face_hash,
                embedding_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        values = []
        for e in embeddings:
            face = e["face"]
            face_shape = list(face.shape)
            face_bytes = face.astype(np.float32).tobytes()
            face_json = json.dumps(face.tolist())

            embedding_bytes = struct.pack(f'{len(e["embedding"])}d', *e["embedding"])

            # uniqueness is guaranteed by face hash and embedding hash
            face_hash = hashlib.sha256(face_json.encode()).hexdigest()
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

            values.append(
                (
                    e["img_name"],
                    face_bytes,
                    face_shape,
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

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:

        query = """
            SELECT id, img_name, embedding
            FROM embeddings
            WHERE model_name = %s AND detector_backend = %s AND aligned = %s AND l2_normalized = %s
            ORDER BY id ASC;
        """

        embeddings: List[Dict[str, Any]] = []

        with self.conn.cursor(name="embeddings_cursor") as cur:
            cur.execute(query, (model_name, detector_backend, aligned, l2_normalized))
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break

                for r in batch:
                    embeddings.append(
                        {
                            "id": r[0],
                            "img_name": r[1],
                            "embedding": r[2],
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
        raise NotImplementedError("ANN search is not natively supported in Postgres.")
