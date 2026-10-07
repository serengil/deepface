# built-in dependencies
import os
import re
import json
import uuid
import hashlib
import struct
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote
from typing import Any, Dict, Optional, List, Tuple, Union, cast

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.exceptions import DuplicateEntryError
from deepface.commons.logger import Logger

logger = Logger()

_KEYSPACE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,47}$")

_DEFAULT_KEYSPACE = "deepface"

_DEFAULT_REPLICATION = {"class": "SimpleStrategy", "replication_factor": 1}

# embeddings of a criteria are spread over partitions of this many ids,
# keeping partitions bounded (~32 MB for 4096 dimensional embeddings)
_BUCKET_SIZE = 1000

# writes larger than half of commitlog_segment_size (16 MB by default) are rejected,
# so the faiss index is stored in chunks
_INDEX_CHUNK_SIZE = 1024 * 1024

_CONCURRENCY = 32

_SEQUENCE_NAME = "embedding_id"

# {ks} is replaced with the keyspace, which is validated against _KEYSPACE_PATTERN
CREATE_TABLES_CQL = [
    """
    CREATE TABLE IF NOT EXISTS {ks}.embeddings (
        id bigint PRIMARY KEY,
        img_name text,
        face blob,
        face_shape text,
        model_name text,
        detector_backend text,
        aligned boolean,
        l2_normalized boolean,
        embedding blob,
        face_hash text,
        embedding_hash text,
        created_at timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {ks}.embeddings_by_criteria (
        model_name text,
        detector_backend text,
        aligned boolean,
        l2_normalized boolean,
        bucket bigint,
        id bigint,
        img_name text,
        embedding blob,
        PRIMARY KEY ((model_name, detector_backend, aligned, l2_normalized, bucket), id)
    ) WITH CLUSTERING ORDER BY (id ASC)
    """,
    """
    CREATE TABLE IF NOT EXISTS {ks}.embeddings_buckets (
        model_name text,
        detector_backend text,
        aligned boolean,
        l2_normalized boolean,
        bucket bigint,
        PRIMARY KEY ((model_name, detector_backend, aligned, l2_normalized), bucket)
    ) WITH CLUSTERING ORDER BY (bucket ASC)
    """,
    """
    CREATE TABLE IF NOT EXISTS {ks}.embeddings_by_hash (
        face_hash text,
        embedding_hash text,
        id bigint,
        PRIMARY KEY ((face_hash, embedding_hash))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {ks}.sequences (
        name text PRIMARY KEY,
        next_id bigint
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {ks}.embeddings_index (
        model_name text,
        detector_backend text,
        align boolean,
        l2_normalized boolean,
        version uuid,
        chunks int,
        created_at timestamp,
        updated_at timestamp,
        PRIMARY KEY ((model_name, detector_backend, align, l2_normalized))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {ks}.embeddings_index_chunks (
        version uuid,
        chunk int,
        data blob,
        PRIMARY KEY ((version), chunk)
    )
    """,
]


def _decode_embedding(embedding_bytes: bytes) -> List[float]:
    """
    Embeddings are stored as little-endian float64 bytes to keep them lossless.
    """
    return cast(List[float], np.frombuffer(embedding_bytes, dtype="<f8").tolist())


def _parse_uri(uri: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse a uri as cassandra://user:password@host1,host2:9042/keyspace into
        cluster arguments and keyspace.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "cassandra":
        raise ValueError("Cassandra uri must start with cassandra://")

    userinfo, _, hosts = parsed.netloc.rpartition("@")

    details: Dict[str, Any] = {"contact_points": []}
    for host in hosts.split(","):
        host, _, port = host.partition(":")
        details["contact_points"].append(host)
        if port:
            details["port"] = int(port)

    if userinfo:
        username, _, password = userinfo.partition(":")
        details["username"] = unquote(username)
        details["password"] = unquote(password)

    return details, parsed.path.strip("/") or _DEFAULT_KEYSPACE


# pylint: disable=too-many-positional-arguments, too-many-instance-attributes
class CassandraClient(Database):
    """
    Cassandra client for DeepFace embeddings storage. Also compatible with ScyllaDB.
    """

    def __init__(
        self,
        connection_details: Optional[Union[Dict[str, Any], str]] = None,
        connection: Any = None,
    ) -> None:
        # Import here to avoid mandatory dependency
        try:
            from cassandra import ConsistencyLevel
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
            from cassandra.query import SimpleStatement
            from cassandra.concurrent import execute_concurrent_with_args
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "cassandra-driver is an optional dependency, ensure the library is installed."
                "Please install using 'pip install cassandra-driver'"
            ) from e

        self.ConsistencyLevel = ConsistencyLevel
        self.SimpleStatement = SimpleStatement
        self.execute_concurrent_with_args = execute_concurrent_with_args

        replication = _DEFAULT_REPLICATION

        if connection is not None:
            # an existing session, whose keyspace is used if it is set
            self.session = connection
            keyspace = connection.keyspace or _DEFAULT_KEYSPACE
        else:
            # Retrieve connection details from parameter or environment variable
            self.conn_details = connection_details or os.environ.get("DEEPFACE_CASSANDRA_URI")
            if not self.conn_details:
                raise ValueError(
                    "Cassandra connection information not found. "
                    "Please provide connection_details or set the DEEPFACE_CASSANDRA_URI"
                    " environment variable."
                )

            if isinstance(self.conn_details, str):
                cluster_args, keyspace = _parse_uri(self.conn_details)
            elif isinstance(self.conn_details, dict):
                cluster_args = dict(self.conn_details)
                keyspace = cluster_args.pop("keyspace", _DEFAULT_KEYSPACE)
                replication = cluster_args.pop("replication", _DEFAULT_REPLICATION)
            else:
                raise ValueError("connection_details must be either a string or a dict.")

            username = cluster_args.pop("username", None)
            password = cluster_args.pop("password", None)
            if username is not None:
                cluster_args["auth_provider"] = PlainTextAuthProvider(
                    username=username, password=password
                )

            self.session = Cluster(**cluster_args).connect()

        if not _KEYSPACE_PATTERN.match(keyspace):
            raise ValueError(f"Invalid keyspace name {keyspace!r}.")
        self.keyspace = keyspace

        self.initialize_database(replication=replication)

    def _cql(self, query: str) -> str:
        return query.format(ks=self.keyspace)

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure that the keyspace and the tables exist.
        """
        replication = kwargs.get("replication", _DEFAULT_REPLICATION)
        replication_cql = ", ".join(f"'{k}': '{v}'" for k, v in replication.items())
        self.session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {self.keyspace}"
            f" WITH replication = {{{replication_cql}}}"
        )
        for query in CREATE_TABLES_CQL:
            self.session.execute(self._cql(query))
        logger.debug(f"Ensured tables either exist or were created in {self.keyspace} keyspace.")

        self._insert_embedding = self.session.prepare(
            self._cql(
                "INSERT INTO {ks}.embeddings (id, img_name, face, face_shape, model_name,"
                " detector_backend, aligned, l2_normalized, embedding, face_hash,"
                " embedding_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
        )
        self._insert_by_criteria = self.session.prepare(
            self._cql(
                "INSERT INTO {ks}.embeddings_by_criteria (model_name, detector_backend, aligned,"
                " l2_normalized, bucket, id, img_name, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
        )
        self._insert_bucket = self.session.prepare(
            self._cql(
                "INSERT INTO {ks}.embeddings_buckets (model_name, detector_backend, aligned,"
                " l2_normalized, bucket) VALUES (?, ?, ?, ?, ?)"
            )
        )
        self._claim_hash = self.session.prepare(
            self._cql(
                "INSERT INTO {ks}.embeddings_by_hash (face_hash, embedding_hash, id)"
                " VALUES (?, ?, ?) IF NOT EXISTS"
            )
        )
        self._release_hash = self.session.prepare(
            self._cql(
                "DELETE FROM {ks}.embeddings_by_hash WHERE face_hash = ? AND embedding_hash = ?"
                " IF id = ?"
            )
        )
        self._select_by_id = self.session.prepare(
            self._cql(
                "SELECT id, img_name, model_name, detector_backend, aligned, l2_normalized,"
                " embedding FROM {ks}.embeddings WHERE id = ?"
            )
        )
        self._select_img_name = self.session.prepare(
            self._cql("SELECT id, img_name FROM {ks}.embeddings WHERE id = ?")
        )
        self._insert_index_chunk = self.session.prepare(
            self._cql(
                "INSERT INTO {ks}.embeddings_index_chunks (version, chunk, data) VALUES (?, ?, ?)"
            )
        )

    def close(self) -> None:
        """Close the database connection."""
        self.session.cluster.shutdown()

    def _allocate_ids(self, count: int) -> List[int]:
        """
        Reserve a block of ids with a compare-and-set on the sequence row.
        """
        select = self.SimpleStatement(
            self._cql("SELECT next_id FROM {ks}.sequences WHERE name = %s"),
            consistency_level=self.ConsistencyLevel.SERIAL,
        )
        while True:
            row = self.session.execute(select, (_SEQUENCE_NAME,)).one()
            if row is None:
                first_id = 1
                applied = self.session.execute(
                    self._cql(
                        "INSERT INTO {ks}.sequences (name, next_id) VALUES (%s, %s) IF NOT EXISTS"
                    ),
                    (_SEQUENCE_NAME, first_id + count),
                ).was_applied
            else:
                first_id = row.next_id
                applied = self.session.execute(
                    self._cql(
                        "UPDATE {ks}.sequences SET next_id = %s WHERE name = %s IF next_id = %s"
                    ),
                    (first_id + count, _SEQUENCE_NAME, first_id),
                ).was_applied
            if applied:
                return list(range(first_id, first_id + count))

    def _execute_concurrent(self, statement: Any, parameters: List[Tuple[Any, ...]]) -> List[Any]:
        return cast(
            List[Any],
            self.execute_concurrent_with_args(
                self.session,
                statement,
                parameters,
                concurrency=_CONCURRENCY,
                raise_on_first_error=False,
            ),
        )

    def upsert_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        index_data: bytes,
    ) -> None:
        """
        Upsert embeddings index into Cassandra. Chunks of a new version are written first,
            then the pointer is switched to the new version, and the old version is removed.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            index_data (bytes): Serialized index data.
        """
        criteria = (model_name, detector_backend, aligned, l2_normalized)

        previous = self.session.execute(
            self._cql(
                "SELECT version FROM {ks}.embeddings_index WHERE model_name = %s"
                " AND detector_backend = %s AND align = %s AND l2_normalized = %s"
            ),
            criteria,
        ).one()

        version = uuid.uuid4()
        chunks = [
            (version, i, index_data[offset : offset + _INDEX_CHUNK_SIZE])
            for i, offset in enumerate(range(0, len(index_data), _INDEX_CHUNK_SIZE))
        ]
        for success, result in self._execute_concurrent(self._insert_index_chunk, chunks):
            if not success:
                raise result

        now = datetime.now(timezone.utc)
        self.session.execute(
            self._cql(
                "UPDATE {ks}.embeddings_index SET version = %s, chunks = %s, updated_at = %s"
                " WHERE model_name = %s AND detector_backend = %s AND align = %s"
                " AND l2_normalized = %s"
            ),
            (version, len(chunks), now, *criteria),
        )
        if previous is None:
            self.session.execute(
                self._cql(
                    "UPDATE {ks}.embeddings_index SET created_at = %s WHERE model_name = %s"
                    " AND detector_backend = %s AND align = %s AND l2_normalized = %s"
                ),
                (now, *criteria),
            )
        else:
            self.session.execute(
                self._cql("DELETE FROM {ks}.embeddings_index_chunks WHERE version = %s"),
                (previous.version,),
            )

    def get_embeddings_index(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> bytes:
        """
        Get embeddings index from Cassandra.
        Args:
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the embeddings are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
        Returns:
            bytes: Serialized index data.
        """
        pointer = self.session.execute(
            self._cql(
                "SELECT version, chunks FROM {ks}.embeddings_index WHERE model_name = %s"
                " AND detector_backend = %s AND align = %s AND l2_normalized = %s"
            ),
            (model_name, detector_backend, aligned, l2_normalized),
        ).one()

        if pointer is None or pointer.version is None:
            raise ValueError(
                "No Embeddings index found for the specified parameters "
                f" {model_name=}, {detector_backend=}, {aligned=}, {l2_normalized=}. "
                "You must run build_index first."
            )

        statement = self.SimpleStatement(
            self._cql("SELECT data FROM {ks}.embeddings_index_chunks WHERE version = %s"),
            fetch_size=8,
        )
        chunks = [row.data for row in self.session.execute(statement, (pointer.version,))]
        if len(chunks) != pointer.chunks:
            raise ValueError(
                f"Embeddings index is incomplete: found {len(chunks)} of {pointer.chunks} chunks."
            )
        return b"".join(chunks)

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert multiple embeddings into Cassandra.
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
                    "aligned": bool(e["aligned"]),
                    "l2_normalized": bool(e["l2_normalized"]),
                    "embedding": embedding_bytes,
                    "face_hash": face_hash,
                    "embedding_hash": embedding_hash,
                }
            )

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            ids = self._allocate_ids(len(batch))

            # claim hashes, and release the claimed ones if any record is a duplicate
            claims = self._execute_concurrent(
                self._claim_hash,
                [(r["face_hash"], r["embedding_hash"], j) for r, j in zip(batch, ids)],
            )
            claimed = [success and result.was_applied for success, result in claims]

            if not all(claimed):
                self._execute_concurrent(
                    self._release_hash,
                    [
                        (r["face_hash"], r["embedding_hash"], j)
                        for r, j, is_claimed in zip(batch, ids, claimed)
                        if is_claimed
                    ],
                )
                for success, result in claims:
                    if not success:
                        raise result

                if len(records) == 1:
                    logger.warn("Duplicate detected for extracted face and embedding.")
                    return 0
                raise DuplicateEntryError(
                    f"Duplicate detected for extracted face and embedding in {i}-th batch"
                )

            now = datetime.now(timezone.utc)
            criteria = [
                (r["model_name"], r["detector_backend"], r["aligned"], r["l2_normalized"])
                for r in batch
            ]
            writes: List[Tuple[Any, List[Tuple[Any, ...]]]] = [
                (
                    self._insert_embedding,
                    [
                        (
                            j,
                            r["img_name"],
                            r["face"],
                            r["face_shape"],
                            r["model_name"],
                            r["detector_backend"],
                            r["aligned"],
                            r["l2_normalized"],
                            r["embedding"],
                            r["face_hash"],
                            r["embedding_hash"],
                            now,
                        )
                        for r, j in zip(batch, ids)
                    ],
                ),
                (
                    self._insert_by_criteria,
                    [
                        (*c, j // _BUCKET_SIZE, j, r["img_name"], r["embedding"])
                        for r, j, c in zip(batch, ids, criteria)
                    ],
                ),
                (
                    self._insert_bucket,
                    list({(*c, j // _BUCKET_SIZE) for j, c in zip(ids, criteria)}),
                ),
            ]
            for statement, parameters in writes:
                for success, result in self._execute_concurrent(statement, parameters):
                    if not success:
                        raise result

        return len(records)

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        criteria = (model_name, detector_backend, aligned, l2_normalized)

        buckets = self.session.execute(
            self._cql(
                "SELECT bucket FROM {ks}.embeddings_buckets WHERE model_name = %s"
                " AND detector_backend = %s AND aligned = %s AND l2_normalized = %s"
            ),
            criteria,
        )

        statement = self.SimpleStatement(
            self._cql(
                "SELECT id, img_name, embedding FROM {ks}.embeddings_by_criteria"
                " WHERE model_name = %s AND detector_backend = %s AND aligned = %s"
                " AND l2_normalized = %s AND bucket = %s"
            ),
            fetch_size=batch_size,
        )

        embeddings: List[Dict[str, Any]] = []
        for bucket in buckets:
            for r in self.session.execute(statement, (*criteria, bucket.bucket)):
                embeddings.append(
                    {
                        "id": r.id,
                        "img_name": r.img_name,
                        "embedding": _decode_embedding(r.embedding),
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
        Fetch a single embedding record with its vector from Cassandra. Criteria arguments
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
        r = self.session.execute(self._select_by_id, (int(identity_id),)).one()

        if r is None:
            return None

        return {
            "id": r.id,
            "img_name": r.img_name,
            "model_name": r.model_name,
            "detector_backend": r.detector_backend,
            "aligned": r.aligned,
            "l2_normalized": r.l2_normalized,
            "embedding": _decode_embedding(r.embedding),
        }

    def search_by_id(
        self,
        ids: Union[List[str], List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Search records by their IDs. Each id is a partition, so they are queried concurrently
            instead of a multi-partition IN clause.
        """
        if not ids:
            return []

        sorted_ids = sorted(int(identity_id) for identity_id in ids)

        results: List[Dict[str, Any]] = []
        for success, result in self._execute_concurrent(
            self._select_img_name, [(identity_id,) for identity_id in sorted_ids]
        ):
            if not success:
                raise result
            r = result.one()
            if r is not None:
                results.append({"id": r.id, "img_name": r.img_name})

        return results
