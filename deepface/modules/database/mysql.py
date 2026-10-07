# built-in dependencies
import os
import json
import hashlib
import struct
from urllib.parse import urlparse, unquote
from typing import Any, Dict, Optional, List, Union, cast

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.exceptions import DuplicateEntryError
from deepface.commons.logger import Logger

logger = Logger()

# keep the number of placeholders in a single statement moderate
_SEARCH_BY_ID_CHUNK_SIZE = 500

# statements larger than max_allowed_packet (64 MB by default) are rejected,
# so the faiss index is stored in chunks
_INDEX_CHUNK_SIZE = 1024 * 1024

# ER_DUP_ENTRY
_DUPLICATE_ENTRY_CODE = 1062

# ER_TABLEACCESS_DENIED_ERROR
_ACCESS_DENIED_CODE = 1142

# BLOB is limited to 64 KB, MEDIUMBLOB to 16 MB and LONGBLOB to 4 GB
CREATE_EMBEDDINGS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        img_name TEXT NOT NULL,
        face LONGBLOB NOT NULL,
        face_shape VARCHAR(64) NOT NULL,
        model_name VARCHAR(100) NOT NULL,
        detector_backend VARCHAR(100) NOT NULL,
        aligned BOOLEAN DEFAULT TRUE,
        l2_normalized BOOLEAN DEFAULT FALSE,
        embedding MEDIUMBLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        face_hash CHAR(64) NOT NULL,
        embedding_hash CHAR(64) NOT NULL,
        UNIQUE KEY uq_embeddings_face_embedding (face_hash, embedding_hash),
        KEY ix_embeddings_criteria (model_name, detector_backend, aligned, l2_normalized, id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

CREATE_EMBEDDINGS_INDEX_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS embeddings_index (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(100) NOT NULL,
        detector_backend VARCHAR(100) NOT NULL,
        align BOOLEAN NOT NULL,
        l2_normalized BOOLEAN NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_embeddings_index_config (model_name, detector_backend, align, l2_normalized)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

CREATE_EMBEDDINGS_INDEX_CHUNKS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS embeddings_index_chunks (
        index_id INT NOT NULL,
        chunk INT NOT NULL,
        data MEDIUMBLOB NOT NULL,
        PRIMARY KEY (index_id, chunk),
        CONSTRAINT fk_embeddings_index_chunks FOREIGN KEY (index_id)
            REFERENCES embeddings_index (id) ON DELETE CASCADE
    ) ENGINE=InnoDB
"""


def _decode_embedding(embedding_bytes: bytes) -> List[float]:
    """
    Embeddings are stored as little-endian float64 bytes to keep them lossless.
    """
    return cast(List[float], np.frombuffer(embedding_bytes, dtype="<f8").tolist())


def _parse_uri(uri: str) -> Dict[str, Any]:
    """
    Parse a uri as mysql://user:password@host:3306/database into connection arguments.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "mysql":
        raise ValueError("MySQL uri must start with mysql://")

    details: Dict[str, Any] = {"host": parsed.hostname or "localhost"}
    if parsed.port:
        details["port"] = parsed.port
    if parsed.username:
        details["user"] = unquote(parsed.username)
    if parsed.password:
        details["password"] = unquote(parsed.password)
    if parsed.path.strip("/"):
        details["database"] = parsed.path.strip("/")
    return details


def _error_code(error: Exception) -> Optional[int]:
    return error.args[0] if error.args and isinstance(error.args[0], int) else None


# pylint: disable=too-many-positional-arguments
class MySqlClient(Database):
    def __init__(
        self,
        connection_details: Optional[Union[Dict[str, Any], str]] = None,
        connection: Any = None,
    ) -> None:
        # Import here to avoid mandatory dependency
        try:
            import pymysql  # type: ignore[import-untyped]
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "pymysql is an optional dependency, ensure the library is installed."
                "Please install using 'pip install pymysql'"
            ) from e

        self.pymysql = pymysql

        if connection is not None:
            self.conn = connection
        else:
            # Retrieve connection details from parameter or environment variable
            self.conn_details = connection_details or os.environ.get("DEEPFACE_MYSQL_URI")
            if not self.conn_details:
                raise ValueError(
                    "MySQL connection information not found. "
                    "Please provide connection_details or set the DEEPFACE_MYSQL_URI"
                    " environment variable."
                )

            # string is a uri, e.g. mysql://user:password@host:3306/database
            if isinstance(self.conn_details, str):
                self.conn = self.pymysql.connect(**_parse_uri(self.conn_details))
            elif isinstance(self.conn_details, dict):
                self.conn = self.pymysql.connect(**self.conn_details)
            else:
                raise ValueError("connection_details must be either a string or a dict.")

        # Ensure the embeddings table exists
        self.initialize_database()

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure that the `embeddings`, `embeddings_index` and `embeddings_index_chunks`
            tables exist.
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute(CREATE_EMBEDDINGS_TABLE_SQL)
                logger.debug("Ensured 'embeddings' table either exists or was created in MySQL.")

                cur.execute(CREATE_EMBEDDINGS_INDEX_TABLE_SQL)
                cur.execute(CREATE_EMBEDDINGS_INDEX_CHUNKS_TABLE_SQL)
                logger.debug(
                    "Ensured 'embeddings_index' tables either exist or were created in MySQL."
                )
            except self.pymysql.err.OperationalError as e:
                if _error_code(e) == _ACCESS_DENIED_CODE:
                    raise ValueError(
                        "The MySQL user does not have permission to create the required tables"
                        " ('embeddings', 'embeddings_index', 'embeddings_index_chunks')."
                        " Please ask your database administrator to grant CREATE privileges."
                    ) from e
                raise
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
        Upsert embeddings index into MySQL. Chunks are replaced in a single transaction,
            so readers see either the previous or the new index entirely.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            index_data (bytes): Serialized index data.
        """
        upsert_query = """
            INSERT INTO embeddings_index (model_name, detector_backend, align, l2_normalized)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), updated_at = CURRENT_TIMESTAMP
        """
        chunk_query = """
            INSERT INTO embeddings_index_chunks (index_id, chunk, data) VALUES (%s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    upsert_query,
                    (model_name, detector_backend, bool(aligned), bool(l2_normalized)),
                )
                index_id = cur.lastrowid

                cur.execute("DELETE FROM embeddings_index_chunks WHERE index_id = %s", (index_id,))
                for i, offset in enumerate(range(0, len(index_data), _INDEX_CHUNK_SIZE)):
                    cur.execute(
                        chunk_query,
                        (index_id, i, index_data[offset : offset + _INDEX_CHUNK_SIZE]),
                    )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> bytes:
        """
        Get embeddings index from MySQL.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
        Returns:
            bytes: Serialized index data.
        """
        query = """
            SELECT c.data
            FROM embeddings_index i
            JOIN embeddings_index_chunks c ON c.index_id = i.id
            WHERE i.model_name = %s AND i.detector_backend = %s
                AND i.align = %s AND i.l2_normalized = %s
            ORDER BY c.chunk ASC
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query, (model_name, detector_backend, bool(aligned), bool(l2_normalized))
            )
            chunks = [bytes(r[0]) for r in cur.fetchall()]

        if chunks:
            return b"".join(chunks)
        raise ValueError(
            "No Embeddings index found for the specified parameters "
            f" {model_name=}, {detector_backend=}, {aligned=}, {l2_normalized=}. "
            "You must run build_index first."
        )

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert multiple embeddings into MySQL.
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    face_bytes,
                    face_shape,
                    e["model_name"],
                    e["detector_backend"],
                    bool(e["aligned"]),
                    bool(e["l2_normalized"]),
                    embedding_bytes,
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
        except self.pymysql.err.IntegrityError as e:
            self.conn.rollback()
            if _error_code(e) != _DUPLICATE_ENTRY_CODE:
                raise
            if len(values) == 1:
                logger.warn("Duplicate detected for extracted face and embedding.")
                return 0
            raise DuplicateEntryError(
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
            ORDER BY id ASC
        """

        embeddings: List[Dict[str, Any]] = []

        # unbuffered cursor streams rows instead of loading the whole result first
        with self.conn.cursor(self.pymysql.cursors.SSCursor) as cur:
            cur.execute(query, (model_name, detector_backend, bool(aligned), bool(l2_normalized)))
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
        Fetch a single embedding record with its vector from MySQL. Criteria arguments
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
            WHERE id = %s
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (int(identity_id),))
            r = cur.fetchone()

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

        with self.conn.cursor() as cur:
            for i in range(0, len(ids), _SEARCH_BY_ID_CHUNK_SIZE):
                chunk = [int(identity_id) for identity_id in ids[i : i + _SEARCH_BY_ID_CHUNK_SIZE]]
                placeholders = ", ".join("%s" for _ in chunk)
                # we may return the face in the future
                cur.execute(
                    f"SELECT id, img_name FROM embeddings WHERE id IN ({placeholders})"
                    " ORDER BY id ASC",
                    chunk,
                )
                for r in cur.fetchall():
                    results.append(
                        {
                            "id": r[0],
                            "img_name": r[1],
                        }
                    )

        return results
