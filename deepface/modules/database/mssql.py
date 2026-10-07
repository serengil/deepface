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

# sql server accepts at most 2100 parameters in a single statement
_SEARCH_BY_ID_CHUNK_SIZE = 500

# error numbers raised by sql server for unique constraint and unique index violations
_UNIQUE_VIOLATION_CODES = ("2627", "2601")

# hashes are kept as fixed length strings, because MAX types cannot be part of a unique key
CREATE_EMBEDDINGS_TABLE_SQL = """
    IF OBJECT_ID(N'embeddings', N'U') IS NULL
    CREATE TABLE embeddings (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        img_name NVARCHAR(MAX) NOT NULL,
        face VARBINARY(MAX) NOT NULL,
        face_shape VARCHAR(64) NOT NULL,
        model_name NVARCHAR(100) NOT NULL,
        detector_backend NVARCHAR(100) NOT NULL,
        aligned BIT DEFAULT 1,
        l2_normalized BIT DEFAULT 0,
        embedding VARBINARY(MAX) NOT NULL,
        created_at DATETIME2 DEFAULT SYSUTCDATETIME(),
        face_hash CHAR(64) NOT NULL,
        embedding_hash CHAR(64) NOT NULL,
        CONSTRAINT uq_embeddings_face_embedding UNIQUE (face_hash, embedding_hash)
    );
"""

CREATE_EMBEDDINGS_INDEX_TABLE_SQL = """
    IF OBJECT_ID(N'embeddings_index', N'U') IS NULL
    CREATE TABLE embeddings_index (
        id INT IDENTITY(1,1) PRIMARY KEY,
        model_name NVARCHAR(100),
        detector_backend NVARCHAR(100),
        align BIT,
        l2_normalized BIT,
        index_data VARBINARY(MAX),
        created_at DATETIME2 DEFAULT SYSUTCDATETIME(),
        updated_at DATETIME2 DEFAULT SYSUTCDATETIME(),
        CONSTRAINT uq_embeddings_index_config
            UNIQUE (model_name, detector_backend, align, l2_normalized)
    );
"""


def _decode_embedding(embedding_bytes: bytes) -> List[float]:
    """
    Embeddings are stored as little-endian float64 bytes to keep them lossless.
    """
    return cast(List[float], np.frombuffer(embedding_bytes, dtype="<f8").tolist())


def _is_unique_violation(error: Exception) -> bool:
    message = " ".join(str(arg) for arg in error.args)
    return any(code in message for code in _UNIQUE_VIOLATION_CODES)


# pylint: disable=too-many-positional-arguments
class MsSqlClient(Database):
    def __init__(
        self,
        connection_details: Optional[Union[Dict[str, Any], str]] = None,
        connection: Any = None,
    ) -> None:
        # Import here to avoid mandatory dependency
        try:
            import pyodbc
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "pyodbc is an optional dependency, ensure the library is installed."
                "Please install using 'pip install pyodbc' and install the"
                " Microsoft ODBC Driver for SQL Server."
            ) from e

        self.pyodbc = pyodbc

        if connection is not None:
            self.conn = connection
        else:
            # Retrieve connection details from parameter or environment variable
            self.conn_details = connection_details or os.environ.get("DEEPFACE_MSSQL_URI")
            if not self.conn_details:
                raise ValueError(
                    "MS SQL Server connection information not found. "
                    "Please provide connection_details or set the DEEPFACE_MSSQL_URI"
                    " environment variable."
                )

            # string is an odbc connection string, e.g.
            # DRIVER={ODBC Driver 18 for SQL Server};SERVER=host,1433;DATABASE=deepface;UID=;PWD=
            if isinstance(self.conn_details, str):
                self.conn = self.pyodbc.connect(self.conn_details)
            elif isinstance(self.conn_details, dict):
                self.conn = self.pyodbc.connect(**self.conn_details)
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
            logger.debug("Ensured 'embeddings' table either exists or was created in MS SQL.")

            cur.execute(CREATE_EMBEDDINGS_INDEX_TABLE_SQL)
            logger.debug("Ensured 'embeddings_index' table either exists or was created in MS SQL.")
        except self.pyodbc.Error as e:
            # 262: CREATE TABLE permission denied
            if "262" in " ".join(str(arg) for arg in e.args):
                raise ValueError(
                    "The MS SQL Server user does not have permission to create "
                    "the required tables ('embeddings', 'embeddings_index'). "
                    "Please ask your database administrator to grant CREATE TABLE privileges."
                ) from e
            raise
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
        Upsert embeddings index into MS SQL Server.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            index_data (bytes): Serialized index data.
        """
        update_query = """
            UPDATE embeddings_index
            SET index_data = ?, updated_at = SYSUTCDATETIME()
            WHERE model_name = ? AND detector_backend = ? AND align = ? AND l2_normalized = ?
        """
        insert_query = """
            INSERT INTO embeddings_index (model_name, detector_backend, align, l2_normalized, index_data)
            VALUES (?, ?, ?, ?, ?)
        """
        criteria = (model_name, detector_backend, bool(aligned), bool(l2_normalized))
        cur = self.conn.cursor()
        try:
            cur.execute(update_query, (index_data, *criteria))
            if cur.rowcount == 0:
                cur.execute(insert_query, (*criteria, index_data))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
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
        Get embeddings index from MS SQL Server.
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
            cur.execute(query, (model_name, detector_backend, bool(aligned), bool(l2_normalized)))
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
        Insert multiple embeddings into MS SQL Server.
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

        cur = self.conn.cursor()
        try:
            for i in range(0, len(values), batch_size):
                cur.executemany(query, values[i : i + batch_size])
                # commit for every batch
                self.conn.commit()
            return len(values)
        except self.pyodbc.IntegrityError as e:
            self.conn.rollback()
            if not _is_unique_violation(e):
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
            cur.execute(query, (model_name, detector_backend, bool(aligned), bool(l2_normalized)))
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break

                for r in batch:
                    embeddings.append(
                        {
                            "id": int(r[0]),
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
        Fetch a single embedding record with its vector from MS SQL Server. Criteria arguments
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
            cur.execute(query, (int(identity_id),))
            r = cur.fetchone()
        finally:
            cur.close()

        if r is None:
            return None

        return {
            "id": int(r[0]),
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
                chunk = [int(identity_id) for identity_id in ids[i : i + _SEARCH_BY_ID_CHUNK_SIZE]]
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
                            "id": int(r[0]),
                            "img_name": r[1],
                        }
                    )
        finally:
            cur.close()

        return results
