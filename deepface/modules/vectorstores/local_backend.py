# built-in dependencies
from typing import List, Dict, Any, Optional
import numpy as np

# project dependencies
from deepface.modules.vectorstores.base import VectorStore
from deepface.modules.verification import find_distance

class LocalVectorStore(VectorStore):
    """
    Local in-memory vector store implementation.
    Stores embeddings in memory for fast access and search.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # In-memory storage: key is a tuple of (model_name, detector_backend, align, l2_normalize)
        # Value is a dict with 'embeddings': list of vectors, 'metadata': list of dicts, 'ids': list of ints
        self.storage: Dict[tuple, Dict[str, Any]] = {}

    def _get_key(self, model_name: str, detector_backend: str, align: bool, l2_normalize: bool) -> tuple:
        return (model_name, detector_backend, align, l2_normalize)

    def connect(self) -> None:
        pass

    def register(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> List[int]:
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        if key not in self.storage:
            self.storage[key] = {
                'embeddings': [],
                'metadata': [],
                'ids': [],
                'next_id': 0
            }

        store = self.storage[key]
        ids = []
        for emb, meta in zip(embeddings, metadata):
            emb_id = store['next_id']
            store['embeddings'].append(emb)
            store['metadata'].append(meta)
            store['ids'].append(emb_id)
            ids.append(emb_id)
            store['next_id'] += 1

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
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        if key not in self.storage:
            return []

        store = self.storage[key]
        results = []

        for i, emb in enumerate(store['embeddings']):
            distance = find_distance(emb, query_embedding, distance_metric)
            if threshold is None or distance <= threshold:
                results.append({
                    'id': store['ids'][i],
                    'distance': distance,
                    'metadata': store['metadata'][i]
                })

        results.sort(key=lambda x: x['distance'])

        if k is not None:
            results = results[:k]

        return results

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
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        if key not in self.storage:
            return False

        store = self.storage[key]
        if id not in store['ids']:
            return False

        index = store['ids'].index(id)
        if embedding is not None:
            store['embeddings'][index] = embedding
        if metadata is not None:
            store['metadata'][index] = metadata

        return True

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
        Retrieve embeddings for a specific identity.
        """
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        if key not in self.storage:
            return []

        store = self.storage[key]
        results = []

        for i, meta in enumerate(store['metadata']):
            if meta.get('identity') == identity:
                results.append({
                    'id': store['ids'][i],
                    'embedding': store['embeddings'][i],
                    'metadata': meta
                })
                if limit is not None and len(results) >= limit:
                    break

        return results

    def delete(
        self,
        ids: List[int],
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> bool:
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        if key not in self.storage:
            return False

        store = self.storage[key]
        ids_set = set(ids)
        new_embeddings = []
        new_metadata = []
        new_ids = []

        for i, emb_id in enumerate(store['ids']):
            if emb_id not in ids_set:
                new_embeddings.append(store['embeddings'][i])
                new_metadata.append(store['metadata'][i])
                new_ids.append(emb_id)

        store['embeddings'] = new_embeddings
        store['metadata'] = new_metadata
        store['ids'] = new_ids

        return True

    def count(
        self,
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> int:
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        if key not in self.storage:
            return 0
        return len(self.storage[key]['embeddings'])

    def close(self) -> None:
        pass