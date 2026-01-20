# built-in dependencies
import os
import json
import struct
import hashlib
from typing import Any, Dict, Optional, List, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules.modeling import build_model
from deepface.modules.database.types import Database
from deepface.modules.exceptions import DuplicateEntryError
from deepface.commons.logger import Logger

logger = Logger()


_SCHEMA_CHECKED: Dict[str, bool] = {}


# pylint: disable=too-many-positional-arguments
class PGVectorClient(Database):
    def __init__(
        self,
        connection_details: Optional[Union[Dict[str, Any], str]] = None,
        connection: Any = None,
    ) -> None:
        # Import here to avoid mandatory dependency
        try:
            import psycopg
            from psycopg import errors
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "psycopg is an optional dependency, ensure the library is installed."
                "Please install using 'pip install \"psycopg[binary]\"' "
            ) from e

        try:
            from pgvector.psycopg import register_vector
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "pgvector is an optional dependency, ensure the library is installed."
                "Please install using 'pip install pgvector' "
            ) from e

        self.psycopg = psycopg
        self.errors = errors

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
            else:
                raise ValueError("connection_details must be either a string or a dict.")

        # is pgvector extension installed?
        try:
            if not _SCHEMA_CHECKED.get("pgvector_extension"):
                with self.conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    self.conn.commit()
                    _SCHEMA_CHECKED["pgvector_extension"] = True
        except Exception as e:
            raise ValueError(
                "Ensure pgvector extension is installed properly by running: "
                "'CREATE EXTENSION IF NOT EXISTS vector;'"
            ) from e

        register_vector(self.conn)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Initialize PostgreSQL database schema for storing embeddings.
        Args:
            model_name (str): Name of the facial recognition model.
            detector_backend (str): Name of the face detector backend.
            aligned (bool): Whether the faces are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
        """
        model_name = kwargs.get("model_name", "VGG-Face")
        detector_backend = kwargs.get("detector_backend", "opencv")
        aligned = kwargs.get("aligned", True)
        l2_normalized = kwargs.get("l2_normalized", False)

        model = build_model(task="facial_recognition", model_name=model_name)
        dimensions = model.output_shape

        table_name = self.__generate_table_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        if _SCHEMA_CHECKED.get(table_name):
            logger.debug("PostgreSQL schema already checked, skipping.")
            return

        create_table_stmt = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                img_name TEXT NOT NULL,
                face BYTEA NOT NULL,
                face_shape INT[] NOT NULL,
                model_name TEXT NOT NULL,
                detector_backend TEXT NOT NULL,
                aligned BOOLEAN DEFAULT true,
                l2_normalized BOOLEAN DEFAULT false,
                embedding vector({dimensions}) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                face_hash TEXT NOT NULL,
                embedding_hash TEXT NOT NULL,
                UNIQUE (face_hash, embedding_hash)
            );
        """

        index_name = f"{table_name}_{'cosine' if l2_normalized else 'euclidean'}_idx"

        create_index_stmt = f"""
            CREATE INDEX {index_name}
            ON {table_name}
            USING hnsw (embedding {'vector_cosine_ops' if l2_normalized else 'vector_l2_ops'});
        """

        try:
            with self.conn.cursor() as cur:
                # create table if not exists
                cur.execute(create_table_stmt)

                # create index if not exists
                try:
                    cur.execute(create_index_stmt)
                except Exception as e:  # pylint: disable=broad-except
                    # unfortunately if not exists is not supported for index creation
                    if getattr(e, "sqlstate", None) == "42P07":
                        self.conn.rollback()
                    else:
                        raise

            self.conn.commit()
        except Exception as e:
            if getattr(e, "sqlstate", None) == "42501":  # permission denied
                raise ValueError(
                    "The PostgreSQL user does not have permission to create "
                    f"the required table {table_name}. "
                    "Please ask your database administrator to grant CREATE privileges "
                    "on the schema."
                ) from e
            raise

        _SCHEMA_CHECKED[table_name] = True

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

        self.initialize_database(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )

        table_name = self.__generate_table_name(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )

        query = f"""
            INSERT INTO {table_name} (
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
            self.conn.rollback()
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
        """
        Fetch all embeddings from PostgreSQL.
        Args:
            model_name (str): Name of the facial recognition model.
            detector_backend (str): Name of the face detector backend.
            aligned (bool): Whether the faces are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            batch_size (int): Number of embeddings to fetch per batch.
        Returns:
            List[Dict[str, Any]]: List of embeddings.
        """

        table_name = self.__generate_table_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        query = f"""
            SELECT id, img_name, embedding
            FROM {table_name}
            ORDER BY id ASC;
        """

        embeddings: List[Dict[str, Any]] = []

        with self.conn.cursor(name="embeddings_cursor") as cur:
            cur.execute(query)
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break

                for r in batch:
                    embeddings.append(
                        {
                            "id": r[0],
                            "img_name": r[1],
                            "embedding": list(r[2]),
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
        Search for similar embeddings in PostgreSQL using a given vector.
        Args:
            vector (List[float]): The query embedding vector.
            model_name (str): Name of the facial recognition model.
            detector_backend (str): Name of the face detector backend.
            aligned (bool): Whether the faces are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            limit (int): Number of similar embeddings to return.
        Returns:
            List[Dict[str, Any]]: List of similar embeddings with distances.
        """
        table_name = self.__generate_table_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        op = "<=>" if l2_normalized else "<->"

        query = f"""
            SELECT id, img_name, (embedding {op} (%s::vector)) AS distance
            FROM {table_name}
            ORDER BY embedding {op} (%s::vector)
            LIMIT %s;
        """

        results: List[Dict[str, Any]] = []

        with self.conn.cursor() as cur:
            cur.execute(query, (vector, vector, limit))
            rows = cur.fetchall()
            for r in rows:
                results.append(
                    {
                        "id": r[0],
                        "img_name": r[1],
                        # "embedding": list(r[2]), # embedding dropped in select query
                        "distance": r[2],
                        "model_name": model_name,
                        "detector_backend": detector_backend,
                        "aligned": aligned,
                        "l2_normalized": l2_normalized,
                    }
                )
        return results

    @staticmethod
    def __generate_table_name(
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> str:
        """
        Generate postgres table name based on parameters.
        """
        class_name_attributes = [
            model_name.replace("-", ""),
            detector_backend,
            "Aligned" if aligned else "Unaligned",
            "Norm" if l2_normalized else "Raw",
        ]
        return "Embeddings_" + "_".join(class_name_attributes).lower()
