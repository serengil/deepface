# built-in dependencies
from typing import Any, Dict, List, Optional, Union
import json

# 3rd party dependencies
try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    raise ImportError("pymilvus is required for Milvus backend. Install it with: pip install pymilvus")

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.verification import find_distance
from deepface.commons.logger import Logger

logger = Logger()


class MilvusClientDB(Database):
    """
    Milvus vector database client implementing the Database interface.
    """

    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        # MilvusClientDB initialization
        if connection is not None:
            self.client = connection
            self.uri = getattr(connection, "uri", str(id(connection)))
        else:
            if isinstance(connection_details, str):
                self.uri = connection_details
            elif isinstance(connection_details, dict):
                self.uri = connection_details.get("uri", "http://localhost:19530")
            else:
                self.uri = "http://localhost:19530"
            self.client = MilvusClient(uri=self.uri)
        self.collection_prefix = "deepface"

    def _get_collection_name(
        self, model_name: str, detector_backend: str, align: bool, l2_normalize: bool
    ) -> str:
        align_str = "aligned" if align else "unaligned"
        l2_str = "l2" if l2_normalize else "nol2"
        # Sanitize model name (replace hyphens)
        model_safe = model_name.replace("-", "")
        return f"{self.collection_prefix}_{model_safe}_{detector_backend}_{align_str}_{l2_str}"

    def initialize_database(self, **kwargs: Any) -> None:
        # Milvus collections are created on first insert; no action needed here.
        pass

    def _ensure_collection(self, collection_name: str, dim: int, l2_normalized: bool = False):
        if not self.client.has_collection(collection_name):
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE" if l2_normalized else "L2",
                params={"M": 16, "efConstruction": 256},
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"Created Milvus collection: {collection_name}")

    def insert_embeddings(
        self, embeddings: List[Dict[str, Any]], batch_size: int = 100
    ) -> int:
        if not embeddings:
            return 0

        # Determine collection name from first record (assume all same)
        first = embeddings[0]
        model_name = first.get("model_name", "VGG-Face")
        detector_backend = first.get("detector_backend", "opencv")
        aligned = first.get("aligned", True)
        l2_normalized = first.get("l2_normalized", False)
        collection_name = self._get_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )

        dim = len(first["embedding"])
        self._ensure_collection(collection_name, dim, l2_normalized)

        data = []
        inserted_count = 0
        for record in embeddings:
            # Generate deterministic ID if not provided
            rec_id = record.get("id")
            if rec_id is None:
                # Deterministic based on img_name and position
                img_name = record.get("img_name", "")
                rec_id = hash(f"{img_name}_{inserted_count}") % (2**63)

            milvus_record = {
                "id": int(rec_id),
                "embedding": record["embedding"],
                "img_name": record.get("img_name", ""),
                "model_name": model_name,
                "detector_backend": detector_backend,
                "aligned": aligned,
                "l2_normalized": l2_normalized,
            }
            # include any additional fields (like identity) - we'll treat as dynamic
            for k, v in record.items():
                if k not in milvus_record and k not in ("face", "id", "embedding"):
                    # ensure value is JSON serializable
                    try:
                        json.dumps(v)
                        milvus_record[k] = v
                    except Exception:
                        pass  # skip non-serializable

            data.append(milvus_record)
            inserted_count += 1

        # Insert in batches (Milvus can handle all at once but we chunk)
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            self.client.insert(collection_name=collection_name, data=batch)

        logger.info(f"Inserted {inserted_count} embeddings into {collection_name}")
        return inserted_count

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        collection_name = self._get_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )
        if not self.client.has_collection(collection_name):
            return []

        # Milvus doesn't support batching natively in query, but we can retrieve all with offset; for simplicity fetch all
        results = self.client.query(
            collection_name=collection_name,
            output_fields=["id", "img_name", "embedding"],
        )
        embeddings = []
        for r in results:
            embeddings.append(
                {
                    "id": r["id"],
                    "img_name": r.get("img_name"),
                    "embedding": r["embedding"],
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
        collection_name = self._get_collection_name(
            model_name, detector_backend, aligned, l2_normalized
        )
        if not self.client.has_collection(collection_name):
            return []

        metric_map = {
            "cosine": "COSINE",
            "euclidean": "L2",
            "euclidean_l2": "L2",
            "angular": "IP",
        }
        metric_type = metric_map.get("cosine", "COSINE")  # default cosine
        search_params = {"metric_type": metric_type, "params": {}}

        res = self.client.search(
            collection_name=collection_name,
            data=[vector],
            anns_field="embedding",
            search_params=search_params,
            limit=limit,
            output_fields=["id", "img_name"],
        )
        results = []
        for hits in res:
            for hit in hits:
                distance = hit["distance"]
                results.append(
                    {
                        "id": hit["id"],
                        "img_name": hit.get("img_name"),
                        "embedding": None,  # not returned by search
                        "distance": distance,
                    }
                )
        return results

    def search_by_identity(
        self,
        identity: str,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        align: bool = True,
        l2_normalize: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        collection_name = self._get_collection_name(
            model_name, detector_backend, align, l2_normalize
        )
        if not self.client.has_collection(collection_name):
            return []

        filter_expr = f'img_name == "{identity}"'
        # Use a high limit if not specified to fetch all matches
        query_limit = limit if limit is not None else 10000
        results = self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=["id", "embedding", "img_name", "model_name", "detector_backend", "aligned", "l2_normalized"],
            limit=query_limit,
        )
        if not self.client.has_collection(collection_name):
            return []

        query_limit = limit if limit is not None else 100
        # Use filter on img_name
        filter_expr = f'img_name == "{identity}"'
        results = self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=["id", "embedding", "img_name", "model_name", "detector_backend", "aligned", "l2_normalized"],
            limit=query_limit,
        )
        output = []
        for r in results:
            meta = {
                "img_name": r.get("img_name"),
                "model_name": r.get("model_name"),
                "detector_backend": r.get("detector_backend"),
                "aligned": r.get("aligned"),
                "l2_normalized": r.get("l2_normalized"),
            }
            # Include any other fields stored dynamically
            for k, v in r.items():
                if k not in ("id", "embedding") and k not in meta:
                    meta[k] = v
            output.append(
                {
                    "id": r["id"],
                    "embedding": r["embedding"],
                    "metadata": meta,
                }
            )
        return output

    def count(
        self,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        aligned: Optional[bool] = None,
        l2_normalized: Optional[bool] = None,
    ) -> int:
        if (
            model_name is not None
            and detector_backend is not None
            and aligned is not None
            and l2_normalized is not None
        ):
            collection_name = self._get_collection_name(
                model_name, detector_backend, aligned, l2_normalized
            )
            if not self.client.has_collection(collection_name):
                return 0
            stats = self.client.get_collection_stats(collection_name)
            return stats.get("row_count", 0)
        else:
            # Count all collections
            total = 0
            collections = self.client.list_collections()
            for coll in collections:
                stats = self.client.get_collection_stats(coll)
                total += stats.get("row_count", 0)
            return total

    def delete(
        self,
        ids: List[int],
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        align: bool = True,
        l2_normalize: bool = False,
    ) -> bool:
        if model_name is None or detector_backend is None:
            return False
        collection_name = self._get_collection_name(
            model_name, detector_backend, align, l2_normalize
        )
        if not self.client.has_collection(collection_name):
            return False
        # Milvus delete expects list of ids
        self.client.delete(collection_name=collection_name, ids=ids)
        logger.info(f"Deleted {len(ids)} embeddings from {collection_name}")
        return True

    def update(
        self,
        id: int,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: str = "",
        detector_backend: str = "",
        align: bool = True,
        l2_normalize: bool = False,
    ) -> bool:
        if not model_name or not detector_backend:
            return False
        collection_name = self._get_collection_name(
            model_name, detector_backend, align, l2_normalize
        )
        if not self.client.has_collection(collection_name):
            return False

        update_data = {"id": id}
        if embedding is not None:
            update_data["embedding"] = embedding
        if metadata is not None:
            for k, v in metadata.items():
                if k not in ("id", "embedding"):
                    update_data[k] = v

        # Milvus upsert
        self.client.upsert(collection_name=collection_name, data=[update_data])
        return True

    def close(self) -> None:
        if self.client:
            self.client.close()
