"""Persisted search indexes should contain exactly the current gallery IDs."""

from unittest.mock import Mock

import numpy as np
import pytest

from deepface.modules import datastore

faiss = pytest.importorskip("faiss")


@pytest.fixture
def gallery(monkeypatch):
    db = Mock()
    db.get_embeddings_index.side_effect = ValueError("no saved index")
    monkeypatch.setattr(datastore, "__connect_database", lambda **unused: db)
    return db


def build(gallery, ids, batch_size=2):
    gallery.fetch_all_embeddings.return_value = [
        {"id": i, "embedding": [float(i), 1.0]} for i in ids
    ]
    datastore.build_index(database_type="sqlite", batch_size=batch_size)
    content = gallery.upsert_embeddings_index.call_args.kwargs["index_data"]
    gallery.get_embeddings_index.side_effect = None
    gallery.get_embeddings_index.return_value = content
    index = faiss.deserialize_index(np.frombuffer(content, dtype=np.uint8))
    return faiss.vector_to_array(index.id_map).tolist()


def test_removed_id_is_not_returned_by_rebuilt_index(gallery):
    assert build(gallery, [1, 2, 3]) == [1, 2, 3]
    assert build(gallery, [2, 3]) == [2, 3]
    assert build(gallery, []) == []


def test_index_persistence_does_not_require_a_shared_scratch_file(gallery, monkeypatch):
    original_write = faiss.write_index

    def memory_only(index, target, *args):
        if isinstance(target, str):
            raise PermissionError("no writable scratch")
        return original_write(index, target, *args)

    monkeypatch.setattr(faiss, "write_index", memory_only)
    assert build(gallery, [4, 6]) == [4, 6]


def test_build_stages_at_most_one_batch(gallery, monkeypatch):
    staged = []

    def record(original):
        def checked(data, *args, **kwargs):
            if isinstance(data, list) and data and isinstance(data[0], list):
                staged.append(len(data))
            return original(data, *args, **kwargs)

        return checked

    for name in ("array", "asarray"):
        monkeypatch.setattr(np, name, record(getattr(np, name)))
    assert build(gallery, range(9), batch_size=2) == list(range(9))
    assert staged and max(staged) <= 2
