# built-in dependencies
import os
import re
import json
import hashlib
import struct
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Union, cast

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.exceptions import DuplicateEntryError
from deepface.commons.logger import Logger

logger = Logger()

_KEY_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9_\-.]+$")


def _decode_embedding(embedding_bytes: bytes) -> List[float]:
    """
    Embeddings are stored as little-endian float64 bytes to keep them lossless.
    """
    return cast(List[float], np.frombuffer(embedding_bytes, dtype="<f8").tolist())


def _to_str(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


# pylint: disable=too-many-positional-arguments
class RedisClient(Database):
    """
    Redis client for DeepFace embeddings storage. Keys are laid out as follows:
        {prefix}:embedding_id                        -> counter allocating integer ids
        {prefix}:embedding:{id}                      -> hash storing a single record
        {prefix}:criteria:{model}:{detector}:{a}:{n} -> sorted set of ids per criteria
        {prefix}:dedup:{face_hash}:{embedding_hash}  -> id, guaranteeing uniqueness
        {prefix}:index:{model}:{detector}:{a}:{n}    -> hash storing the faiss index
    Faces and embeddings are binary, so the connection must not set decode_responses.
    """

    def __init__(
        self,
        connection_details: Optional[Union[Dict[str, Any], str]] = None,
        connection: Any = None,
        key_prefix: str = "deepface",
    ) -> None:
        # Import here to avoid mandatory dependency
        try:
            import redis  # type: ignore[import-untyped]
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "redis is an optional dependency, ensure the library is installed."
                "Please install using 'pip install redis'"
            ) from e

        self.redis = redis

        if not _KEY_PREFIX_PATTERN.match(key_prefix):
            raise ValueError(f"Invalid key prefix {key_prefix!r}.")
        self.key_prefix = key_prefix

        if connection is not None:
            self.conn = connection
        else:
            # Retrieve connection details from parameter or environment variable
            self.conn_details = connection_details or os.environ.get("DEEPFACE_REDIS_URI")
            if not self.conn_details:
                raise ValueError(
                    "Redis connection information not found. "
                    "Please provide connection_details or set the DEEPFACE_REDIS_URI"
                    " environment variable."
                )

            # string is a redis url, e.g. redis://user:password@host:6379/0
            if isinstance(self.conn_details, str):
                self.conn = self.redis.Redis.from_url(self.conn_details)
            elif isinstance(self.conn_details, dict):
                self.conn = self.redis.Redis(**self.conn_details)
            else:
                raise ValueError("connection_details must be either a string or a dict.")

        self.initialize_database()

    def _criteria_suffix(
        self, model_name: str, detector_backend: str, aligned: bool, l2_normalized: bool
    ) -> str:
        return f"{model_name}:{detector_backend}:{int(aligned)}:{int(l2_normalized)}"

    def _embedding_key(self, identity_id: Union[str, int]) -> str:
        return f"{self.key_prefix}:embedding:{identity_id}"

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Redis is schemaless, so only the connection is validated.
        """
        self.conn.ping()
        logger.debug("Connected to Redis.")

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
        Upsert embeddings index into Redis.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            index_data (bytes): Serialized index data.
        """
        key = f"{self.key_prefix}:index:" + self._criteria_suffix(
            model_name, detector_backend, aligned, l2_normalized
        )
        now = datetime.now(timezone.utc).isoformat()
        pipe = self.conn.pipeline(transaction=True)
        pipe.hset(key, mapping={"index_data": index_data, "updated_at": now})
        pipe.hsetnx(key, "created_at", now)
        pipe.execute()

    def get_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> bytes:
        """
        Get embeddings index from Redis.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
        Returns:
            bytes: Serialized index data.
        """
        key = f"{self.key_prefix}:index:" + self._criteria_suffix(
            model_name, detector_backend, aligned, l2_normalized
        )
        index_data = self.conn.hget(key, "index_data")
        if index_data is not None:
            return bytes(index_data)
        raise ValueError(
            "No Embeddings index found for the specified parameters "
            f" {model_name=}, {detector_backend=}, {aligned=}, {l2_normalized=}. "
            "You must run build_index first."
        )

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert multiple embeddings into Redis.
        Args:
            embeddings (List[Dict[str, Any]]): List of embeddings to insert.
            batch_size (int): Number of embeddings to insert per batch.
        Returns:
            int: Number of embeddings inserted.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        records = []
        for e in embeddings:
            face = e["face"]
            face_json = json.dumps(face.tolist())

            # the same bytes are stored and hashed
            embedding_bytes = struct.pack(f'<{len(e["embedding"])}d', *e["embedding"])

            # uniqueness is guaranteed by face hash and embedding hash
            face_hash = hashlib.sha256(face_json.encode()).hexdigest()
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

            records.append(
                {
                    "img_name": e["img_name"],
                    "face": face.astype(np.float32).tobytes(),
                    "face_shape": json.dumps(list(face.shape)),
                    "model_name": e["model_name"],
                    "detector_backend": e["detector_backend"],
                    "aligned": int(e["aligned"]),
                    "l2_normalized": int(e["l2_normalized"]),
                    "embedding": embedding_bytes,
                    "face_hash": face_hash,
                    "embedding_hash": embedding_hash,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            # reserve a block of ids for the batch
            last_id = self.conn.incrby(f"{self.key_prefix}:embedding_id", len(batch))
            ids = list(range(last_id - len(batch) + 1, last_id + 1))

            # claim dedup keys atomically, and release them if any record is a duplicate
            dedup_keys = [
                f"{self.key_prefix}:dedup:{r['face_hash']}:{r['embedding_hash']}" for r in batch
            ]
            pipe = self.conn.pipeline(transaction=False)
            for dedup_key, identity_id in zip(dedup_keys, ids):
                pipe.set(dedup_key, identity_id, nx=True)
            claimed = pipe.execute()

            if not all(claimed):
                pipe = self.conn.pipeline(transaction=False)
                for dedup_key, is_claimed in zip(dedup_keys, claimed):
                    if is_claimed:
                        pipe.delete(dedup_key)
                pipe.execute()

                if len(records) == 1:
                    logger.warn("Duplicate detected for extracted face and embedding.")
                    return 0
                raise DuplicateEntryError(
                    f"Duplicate detected for extracted face and embedding in {i}-th batch"
                )

            pipe = self.conn.pipeline(transaction=True)
            for identity_id, r in zip(ids, batch):
                pipe.hset(self._embedding_key(identity_id), mapping=r)
                criteria_key = f"{self.key_prefix}:criteria:" + self._criteria_suffix(
                    r["model_name"], r["detector_backend"], r["aligned"], r["l2_normalized"]
                )
                pipe.zadd(criteria_key, {str(identity_id): identity_id})
            pipe.execute()

        return len(records)

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        criteria_key = f"{self.key_prefix}:criteria:" + self._criteria_suffix(
            model_name, detector_backend, aligned, l2_normalized
        )

        embeddings: List[Dict[str, Any]] = []

        start = 0
        while True:
            ids = self.conn.zrange(criteria_key, start, start + batch_size - 1)
            if not ids:
                break
            start += len(ids)

            pipe = self.conn.pipeline(transaction=False)
            for identity_id in ids:
                pipe.hmget(self._embedding_key(_to_str(identity_id)), ["img_name", "embedding"])

            for identity_id, (img_name, embedding_bytes) in zip(ids, pipe.execute()):
                if embedding_bytes is None:
                    continue
                embeddings.append(
                    {
                        "id": int(identity_id),
                        "img_name": _to_str(img_name),
                        "embedding": _decode_embedding(embedding_bytes),
                        "model_name": model_name,
                        "detector_backend": detector_backend,
                        "aligned": aligned,
                        "l2_normalized": l2_normalized,
                    }
                )
        return embeddings

    # criteria arguments are not required, because the record is located by its id only
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
        Fetch a single embedding record with its vector from Redis. Criteria arguments
            are ignored, because the record is located by its id only, and the record
            itself carries the criteria that it was registered with.
        Args:
            identity_id (str or int): ID of the record to fetch.
            model_name (str): Name of the model. Ignored.
            detector_backend (str): Name of the detector backend. Ignored.
            aligned (bool): Whether the embeddings are aligned. Ignored.
            l2_normalized (bool): Whether the embeddings are L2 normalized. Ignored.
        Returns:
            Optional[Dict[str, Any]]: Embedding record, or None if no record found for given id.
        """
        r = self.conn.hmget(
            self._embedding_key(int(identity_id)),
            ["img_name", "model_name", "detector_backend", "aligned", "l2_normalized", "embedding"],
        )

        if r[5] is None:
            return None

        return {
            "id": int(identity_id),
            "img_name": _to_str(r[0]),
            "model_name": _to_str(r[1]),
            "detector_backend": _to_str(r[2]),
            "aligned": bool(int(r[3])),
            "l2_normalized": bool(int(r[4])),
            "embedding": _decode_embedding(r[5]),
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

        sorted_ids = sorted(int(identity_id) for identity_id in ids)

        pipe = self.conn.pipeline(transaction=False)
        for identity_id in sorted_ids:
            pipe.hget(self._embedding_key(identity_id), "img_name")

        results: List[Dict[str, Any]] = []
        for identity_id, img_name in zip(sorted_ids, pipe.execute()):
            if img_name is None:
                continue
            results.append(
                {
                    "id": identity_id,
                    "img_name": _to_str(img_name),
                }
            )

        return results
