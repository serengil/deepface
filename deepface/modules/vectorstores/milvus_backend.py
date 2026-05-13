# built-in dependencies
from typing import List, Dict, Any, Optional
import json

# 3rd party dependencies
try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    raise ImportError("pymilvus is required for Milvus backend. Install it with: pip install pymilvus")

# project dependencies
from deepface.modules.vectorstores.base import VectorStore
from deepface.commons.logger import Logger

logger = Logger()


class MilvusVectorStore(VectorStore):
    """
    Milvus vector store implementation for scalable vector search.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.uri = self.config.get('uri', 'http://localhost:19530')
        self.collection_prefix = self.config.get('collection_prefix', 'deepface')
        self.client = None

    def connect(self) -> None:
        if self.client is None:
            self.client = MilvusClient(uri=self.uri)

    def _get_collection_name(self, model_name: str, detector_backend: str, align: bool, l2_normalize: bool) -> str:
        align_str = 'aligned' if align else 'unaligned'
        l2_str = 'l2' if l2_normalize else 'nol2'
        return f"{self.collection_prefix}_{model_name}_{detector_backend}_{align_str}_{l2_str}"

    def _ensure_collection(self, collection_name: str, dim: int, l2_normalized: bool = False):
        if not self.client.has_collection(collection_name):
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE" if l2_normalized else "L2",
                params={"M": 16, "efConstruction": 256}
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"Created Milvus collection: {collection_name}")

    def register(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> List[int]:
        collection_name = self._get_collection_name(model_name, detector_backend, align, l2_normalize)
        dim = len(embeddings[0]) if embeddings else 0
        self._ensure_collection(collection_name, dim, l2_normalize)

        data = []
        ids = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            # Generate deterministic ID
            emb_id = hash(str(meta.get('img_name', '')) + str(i)) % (2**63)
            ids.append(emb_id)
            record = {
                'id': emb_id,
                'embedding': emb,
                **meta  # Dynamic fields including identity
            }
            data.append(record)

        self.client.insert(collection_name=collection_name, data=data)
        logger.info(f"Inserted {len(data)} embeddings into {collection_name}")
        return ids

    def search(
        self,
        query_embedding: List[float],
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        distance_metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        collection_name = self._get_collection_name(model_name, detector_backend, align, l2_normalize)
        if not self.client.has_collection(collection_name):
            return []

        metric_map = {'cosine': 'COSINE', 'euclidean': 'L2', 'euclidean_l2': 'L2', 'angular': 'IP'}
        milvus_metric = metric_map.get(distance_metric, 'COSINE')

        search_params = {"metric_type": milvus_metric, "params": {}}

        results = self.client.search(
            collection_name=collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=k or 10,
            output_fields=["*"]
        )

        search_results = []
        for hit in results[0]:
            distance = hit['distance']
            if threshold is None or distance <= threshold:
                search_results.append({
                    'id': hit['id'],
                    'distance': distance,
                    'metadata': {k: v for k, v in hit.items() if k not in ['id', 'embedding', 'distance']}
                })

        return search_results

    def search_by_identity(
        self,
        identity: str,
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all embeddings for a given identity using metadata filter.
        """
        collection_name = self._get_collection_name(model_name, detector_backend, align, l2_normalize)
        if not self.client.has_collection(collection_name):
            return []

        # Use Milvus query to filter by identity
        filter_expr = f'identity == "{identity}"'
        results = self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=["id", "embedding", "identity", "img_name", "model_name", "detector_backend", "aligned", "l2_normalized"],
            limit=limit or 100
        )

        return [
            {
                'id': r['id'],
                'embedding': r['embedding'],
                'metadata': {k: v for k, v in r.items() if k not in ['id', 'embedding']}
            }
            for r in results
        ]

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
        collection_name = self._get_collection_name(model_name, detector_backend, align, l2_normalize)
        if not self.client.has_collection(collection_name):
            return False

        data = {"id": id}
        if embedding is not None:
            data["embedding"] = embedding
        if metadata is not None:
            data.update(metadata)

        self.client.upsert(collection_name=collection_name, data=[data])
        return True

    def delete(
        self,
        ids: List[int],
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> bool:
        collection_name = self._get_collection_name(model_name, detector_backend, align, l2_normalize)
        if not self.client.has_collection(collection_name):
            return False

        self.client.delete(collection_name=collection_name, ids=ids)
        logger.info(f"Deleted {len(ids)} embeddings from {collection_name}")
        return True

    def count(
        self,
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> int:
        collection_name = self._get_collection_name(model_name, detector_backend, align, l2_normalize)
        if not self.client.has_collection(collection_name):
            return 0
        stats = self.client.get_collection_stats(collection_name)
        return stats.get('row_count', 0)

    def close(self) -> None:
        self.client.close()