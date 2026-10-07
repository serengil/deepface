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
from deepface.modules.exceptions import DuplicateEntryError
from deepface.commons.logger import Logger

logger = Logger()

# sqlite limits the number of host parameters in a single statement
_SEARCH_BY_ID_CHUNK_SIZE = 500

# AUTOINCREMENT prevents reusing ids of deleted rows, which faiss index relies on
CREATE_EMBEDDINGS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        img_name TEXT NOT NULL,
        face BLOB NOT NULL,
        face_shape TEXT NOT NULL,
        model_name TEXT NOT NULL,
        detector_backend TEXT NOT NULL,
        aligned INTEGER DEFAULT 1,
        l2_normalized INTEGER DEFAULT 0,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        face_hash TEXT NOT NULL,
        embedding_hash TEXT NOT NULL,
        UNIQUE (face_hash, embedding_hash)
    );
"""

CREATE_EMBEDDINGS_INDEX_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS embeddings_index (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        detector_backend TEXT,
        align INTEGER,
        l2_normalized INTEGER,
        index_data BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (model_name, detector_backend, align, l2_normalized)
    );
"""


def _decode_embedding(embedding_bytes: bytes) -> List[float]:
    """
    Embeddings are stored as little-endian float64 bytes to keep them lossless.
    """
    return cast(List[float], np.frombuffer(embedding_bytes, dtype="<f8").tolist())


# pylint: disable=too-many-positional-arguments
class SqliteClient(Database):
    def __init__(
        self,
        connection_details: Optional[Union[Dict[str, Any], str]] = None,
        connection: Any = None,
    ) -> None:
        # built-in module, but imported here to be consistent with other clients
        import sqlite3

        self.sqlite3 = sqlite3

        if connection is not None:
            self.conn = connection
        else:
            # Retrieve connection details from parameter or environment variable
            self.conn_details = connection_details or os.environ.get("DEEPFACE_SQLITE_PATH")
            if not self.conn_details:
                raise ValueError(
                    "SQLite connection information not found. "
                    "Please provide connection_details or set the DEEPFACE_SQLITE_PATH"
                    " environment variable."
                )

            if isinstance(self.conn_details, str):
                self.conn = self.sqlite3.connect(self.conn_details)
            elif isinstance(self.conn_details, dict):
                self.conn = self.sqlite3.connect(**self.conn_details)
            else:
                raise ValueError("connection_details must be either a string or a dict.")

        # Ensure the embeddings table exists
        self.initialize_database()

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure that the `embeddings` and `embeddings_index` tables exist.
        """
        cur = self.conn.cursor()
        try:
            cur.execute(CREATE_EMBEDDINGS_TABLE_SQL)
            logger.debug("Ensured 'embeddings' table either exists or was created in SQLite.")

            cur.execute(CREATE_EMBEDDINGS_INDEX_TABLE_SQL)
            logger.debug("Ensured 'embeddings_index' table either exists or was created in SQLite.")
        finally:
            cur.close()
        self.conn.commit()

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
        Upsert embeddings index into SQLite.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            index_data (bytes): Serialized index data.
        """
        query = """
            INSERT INTO embeddings_index (model_name, detector_backend, align, l2_normalized, index_data)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (model_name, detector_backend, align, l2_normalized)
            DO UPDATE SET
                index_data = excluded.index_data,
                updated_at = CURRENT_TIMESTAMP
        """
        cur = self.conn.cursor()
        try:
            cur.execute(
                query,
                (
                    model_name,
                    detector_backend,
                    int(aligned),
                    int(l2_normalized),
                    self.sqlite3.Binary(index_data),
                ),
            )
            self.conn.commit()
        finally:
            cur.close()

    def get_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> bytes:
        """
        Get embeddings index from SQLite.
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
            WHERE model_name = ? AND detector_backend = ? AND align = ? AND l2_normalized = ?
        """
        cur = self.conn.cursor()
        try:
            cur.execute(query, (model_name, detector_backend, int(aligned), int(l2_normalized)))
            result = cur.fetchone()
        finally:
            cur.close()

        if result:
            return bytes(result[0])
        raise ValueError(
            "No Embeddings index found for the specified parameters "
            f" {model_name=}, {detector_backend=}, {aligned=}, {l2_normalized=}. "
            "You must run build_index first."
        )

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert multiple embeddings into SQLite.
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        values = []
        for e in embeddings:
            face = e["face"]
            face_shape = json.dumps(list(face.shape))
            face_bytes = face.astype(np.float32).tobytes()
            face_json = json.dumps(face.tolist())

            # the same bytes are stored and hashed
            embedding_bytes = struct.pack(f'<{len(e["embedding"])}d', *e["embedding"])

            # uniqueness is guaranteed by face hash and embedding hash
            face_hash = hashlib.sha256(face_json.encode()).hexdigest()
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

            values.append(
                (
                    e["img_name"],
                    self.sqlite3.Binary(face_bytes),
                    face_shape,
                    e["model_name"],
                    e["detector_backend"],
                    int(e["aligned"]),
                    int(e["l2_normalized"]),
                    self.sqlite3.Binary(embedding_bytes),
                    face_hash,
                    embedding_hash,
                )
            )

        cur = self.conn.cursor()
        try:
            for i in range(0, len(values), batch_size):
                cur.executemany(query, values[i : i + batch_size])
                # commit for every batch
                self.conn.commit()
            return len(values)
        except self.sqlite3.IntegrityError as e:
            self.conn.rollback()
            if "UNIQUE" not in str(e):
                raise
            if len(values) == 1:
                logger.warn("Duplicate detected for extracted face and embedding.")
                return 0
            raise DuplicateEntryError(
                f"Duplicate detected for extracted face and embedding columns in {i}-th batch"
            ) from e
        finally:
            cur.close()

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
            WHERE model_name = ? AND detector_backend = ? AND aligned = ? AND l2_normalized = ?
            ORDER BY id ASC;
        """

        embeddings: List[Dict[str, Any]] = []

        cur = self.conn.cursor()
        try:
            cur.execute(query, (model_name, detector_backend, int(aligned), int(l2_normalized)))
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break

                for r in batch:
                    embeddings.append(
                        {
                            "id": r[0],
                            "img_name": r[1],
                            "embedding": _decode_embedding(r[2]),
                            "model_name": model_name,
                            "detector_backend": detector_backend,
                            "aligned": aligned,
                            "l2_normalized": l2_normalized,
                        }
                    )
        finally:
            cur.close()
        return embeddings

    # criteria arguments are not required, because all embeddings are stored in a single table
    # pylint: disable=unused-argument
    def fetch_embedding(
        self,
        identity_id: Union[str, int],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        aligned: bool = True,
        l2_normalized: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single embedding record with its vector from SQLite. Criteria arguments
            are ignored, because a single table stores the embeddings of all criteria, and
            the record itself carries the criteria that it was registered with.
        Args:
            identity_id (str or int): ID of the record to fetch.
            model_name (str): Name of the model. Ignored.
            detector_backend (str): Name of the detector backend. Ignored.
            aligned (bool): Whether the embeddings are aligned. Ignored.
            l2_normalized (bool): Whether the embeddings are L2 normalized. Ignored.
        Returns:
            Optional[Dict[str, Any]]: Embedding record, or None if no record found for given id.
        """
        query = """
            SELECT id, img_name, model_name, detector_backend, aligned, l2_normalized, embedding
            FROM embeddings
            WHERE id = ?;
        """

        cur = self.conn.cursor()
        try:
            cur.execute(query, (identity_id,))
            r = cur.fetchone()
        finally:
            cur.close()

        if r is None:
            return None

        return {
            "id": r[0],
            "img_name": r[1],
            "model_name": r[2],
            "detector_backend": r[3],
            "aligned": bool(r[4]),
            "l2_normalized": bool(r[5]),
            "embedding": _decode_embedding(r[6]),
        }

    def search_by_id(
        self,
        ids: Union[List[str], List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Search records by their IDs.
        """
        if not ids:
            return []

        results: List[Dict[str, Any]] = []

        cur = self.conn.cursor()
        try:
            for i in range(0, len(ids), _SEARCH_BY_ID_CHUNK_SIZE):
                chunk = list(ids[i : i + _SEARCH_BY_ID_CHUNK_SIZE])
                placeholders = ", ".join("?" for _ in chunk)
                # we may return the face in the future
                cur.execute(
                    f"SELECT id, img_name FROM embeddings WHERE id IN ({placeholders})"
                    " ORDER BY id ASC;",
                    chunk,
                )
                for r in cur.fetchall():
                    results.append(
                        {
                            "id": r[0],
                            "img_name": r[1],
                        }
                    )
        finally:
            cur.close()

        return results
