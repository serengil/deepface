# built-in dependencies
from unittest.mock import Mock

# 3rd party dependencies
import pytest
import numpy as np

# project dependencies
from deepface.modules import datastore


@pytest.mark.parametrize("caller_owned", [False, True])
@pytest.mark.parametrize("outcome", ["success", "already_indexed", "fetch_failure", "upload_failure"])
def test_index_client_lifetime(monkeypatch, caller_owned, outcome):
    faiss = pytest.importorskip("faiss")
    client = Mock()
    client.fetch_all_embeddings.return_value = [{"id": 17, "embedding": [1.0, 0.0]}]
    monkeypatch.setattr(datastore, "__connect_database", lambda **kwargs: client)
    index = None
    if outcome == "already_indexed":
        index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
        index.add_with_ids(np.array([[1.0, 0.0]], dtype=np.float32), np.array([17]))
    monkeypatch.setattr(datastore, "__get_index", lambda **kwargs: index)
    if outcome == "fetch_failure":
        client.fetch_all_embeddings.side_effect = RuntimeError("database unavailable")
    elif outcome == "upload_failure":
        client.upsert_embeddings_index.side_effect = RuntimeError("database unavailable")

    def build():
        datastore.build_index(database_type="sqlite", connection=object() if caller_owned else None)

    if outcome.endswith("failure"):
        with pytest.raises(RuntimeError, match="database unavailable"):
            build()
    else:
        build()

    assert client.close.call_count == (0 if caller_owned else 1)
