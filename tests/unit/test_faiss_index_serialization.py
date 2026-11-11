# built-in dependencies
import tempfile
from unittest.mock import Mock

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules import datastore


def test_index_build_does_not_overwrite_another_gallery_file(monkeypatch, tmp_path):
    faiss = pytest.importorskip("faiss")
    # Two galleries using the same model and detector previously shared this filename.
    other_index = tmp_path / "VGG-Face_opencv_True_False.faiss"
    other_index.write_bytes(b"another gallery is building its index")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    client = Mock()
    client.fetch_all_embeddings.return_value = [
        {"id": 17, "embedding": [1.0, 0.0]},
        {"id": 28, "embedding": [0.0, 1.0]},
    ]
    monkeypatch.setattr(datastore, "__connect_database", lambda **kwargs: client)
    monkeypatch.setattr(datastore, "__get_index", lambda **kwargs: None)

    datastore.build_index(database_type="sqlite", connection=object(), batch_size=1)

    blob = client.upsert_embeddings_index.call_args.kwargs["index_data"]
    loaded = faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8))
    distances, identities = loaded.search(np.array([[0.0, 1.0]], dtype=np.float32), 1)
    assert identities.tolist() == [[28]]
    assert distances.tolist() == [[0.0]]
    assert loaded.ntotal == 2
    assert other_index.read_bytes() == b"another gallery is building its index"


def test_failed_database_upload_leaves_no_index_file(monkeypatch, tmp_path):
    pytest.importorskip("faiss")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    client = Mock()
    client.fetch_all_embeddings.return_value = [{"id": 17, "embedding": [1.0, 0.0]}]
    client.upsert_embeddings_index.side_effect = RuntimeError("database unavailable")
    monkeypatch.setattr(datastore, "__connect_database", lambda **kwargs: client)
    monkeypatch.setattr(datastore, "__get_index", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="database unavailable"):
        datastore.build_index(database_type="sqlite", connection=object())

    assert list(tmp_path.iterdir()) == []
