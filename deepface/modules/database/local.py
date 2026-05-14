# built-in dependencies
from typing import Any, Dict, List, Optional, Union
import numpy as np

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.verification import find_distance
from deepface.commons.logger import Logger

logger = Logger()


class LocalDatabase(Database):
    """
    In-memory local vector database implementation.
    Stores embeddings in process memory; not persistent across restarts.
    """

    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        super().__init__()
        # Storage key: (model_name, detector_backend, align, l2_normalize) ->
        #   dict with 'embeddings': list, 'metadatas': list, 'ids': list, 'next_id': int
        self.storage: Dict[tuple, Dict[str, Any]] = {}
        # If a pre-built connection/dict is passed, we could initialize storage from it (advanced), but ignore for now.

    def _get_key(
        self,
        model_name: str,
        detector_backend: str,
        align: bool,
        l2_normalize: bool,
    ) -> tuple:
        return (model_name, detector_backend, align, l2_normalize)

    def initialize_database(self, **kwargs: Any) -> None:
        # No initialization needed for local in-memory store
        pass

    def insert_embeddings(
        self, embeddings: List[Dict[str, Any]], batch_size: int = 100
    ) -> int:
        if not embeddings:
            return 0

        count = 0
        for record in embeddings:
            model_name = record.get("model_name", "VGG-Face")
            detector_backend = record.get("detector_backend", "opencv")
            aligned = record.get("aligned", True)
            l2_normalized = record.get("l2_normalized", False)
            key = self._get_key(model_name, detector_backend, aligned, l2_normalized)

            if key not in self.storage:
                self.storage[key] = {
                    "embeddings": [],
                    "metadatas": [],
                    "ids": [],
                    "next_id": 0,
                }

            store = self.storage[key]

            # Assign ID if not provided
            rec_id = record.get("id")
            if rec_id is None:
                rec_id = store["next_id"]
                store["next_id"] += 1

            # Store embedding and metadata (excluding 'face' and large fields not needed for search)
            embedding = record["embedding"]
            # We'll keep minimal metadata: img_name, model_name, detector_backend, aligned, l2_normalized, and any extra
            meta = {
                "img_name": record.get("img_name"),
                "model_name": model_name,
                "detector_backend": detector_backend,
                "aligned": aligned,
                "l2_normalized": l2_normalized,
            }
            # Include any additional fields from record (like extra metadata)
            for k, v in record.items():
                if k not in meta and k not in ("id", "embedding", "face"):
                    meta[k] = v

            store["embeddings"].append(embedding)
            store["metadatas"].append(meta)
            store["ids"].append(rec_id)
            count += 1

        return count

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        key = self._get_key(model_name, detector_backend, aligned, l2_normalized)
        store = self.storage.get(key)
        if not store:
            return []

        embeddings_list = []
        for i, emb in enumerate(store["embeddings"]):
            meta = store["metadatas"][i]
            embeddings_list.append(
                {
                    "id": store["ids"][i],
                    "img_name": meta.get("img_name"),
                    "embedding": emb,
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                    "aligned": aligned,
                    "l2_normalized": l2_normalized,
                }
            )
        return embeddings_list

    def search_by_vector(
        self,
        vector: List[float],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        aligned: bool = True,
        l2_normalized: bool = False,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        key = self._get_key(model_name, detector_backend, aligned, l2_normalized)
        store = self.storage.get(key)
        if not store:
            return []

        results = []
        for idx, emb in enumerate(store["embeddings"]):
            distance = find_distance(emb, vector, distance_metric="cosine")
            meta = store["metadatas"][idx]
            results.append(
                {
                    "id": store["ids"][idx],
                    "img_name": meta.get("img_name"),
                    "embedding": emb,
                    "distance": distance,
                }
            )

        results.sort(key=lambda x: x["distance"])
        if limit is not None:
            results = results[:limit]
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
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        store = self.storage.get(key)
        if not store:
            return []

        results = []
        for i, meta in enumerate(store["metadatas"]):
            if meta.get("img_name") == identity:
                results.append(
                    {
                        "id": store["ids"][i],
                        "embedding": store["embeddings"][i],
                        "metadata": meta,
                    }
                )
                if limit is not None and len(results) >= limit:
                    break
        return results

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
            key = self._get_key(model_name, detector_backend, aligned, l2_normalized)
            store = self.storage.get(key)
            return len(store["embeddings"]) if store else 0
        else:
            # Count all across all keys
            total = 0
            for store in self.storage.values():
                total += len(store["embeddings"])
            return total

    def delete(
        self,
        ids: List[int],
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        align: bool = True,
        l2_normalize: bool = False,
    ) -> bool:
        # If model/detector not specified, we cannot selectively delete; return False
        if model_name is None or detector_backend is None:
            return False
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        store = self.storage.get(key)
        if not store:
            return False

        ids_set = set(ids)
        new_embeddings = []
        new_metadatas = []
        new_ids = []
        for i, eid in enumerate(store["ids"]):
            if eid not in ids_set:
                new_embeddings.append(store["embeddings"][i])
                new_metadatas.append(store["metadatas"][i])
                new_ids.append(eid)
        store["embeddings"] = new_embeddings
        store["metadatas"] = new_metadatas
        store["ids"] = new_ids
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
        if model_name is None or detector_backend is None:
            return False
        key = self._get_key(model_name, detector_backend, align, l2_normalize)
        store = self.storage.get(key)
        if not store:
            return False
        if id not in store["ids"]:
            return False
        idx = store["ids"].index(id)
        if embedding is not None:
            store["embeddings"][idx] = embedding
        if metadata is not None:
            # Merge metadata into existing meta
            meta = store["metadatas"][idx]
            meta.update(metadata)
            store["metadatas"][idx] = meta
        return True

    def close(self) -> None:
        # Nothing to close for in-memory store
        pass
