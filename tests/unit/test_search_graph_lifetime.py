"""Graph search keeps its connection available for background relationship writes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from deepface.modules import datastore


@pytest.mark.parametrize("borrowed", [False, True])
def test_search_does_not_close_graph_client_before_background_linking(monkeypatch, borrowed):
    started, release, completed = Event(), Event(), Event()

    class GraphClient:
        closed = False
        linked = None

        def fetch_all_embeddings(self, **kwargs):
            return [
                {"id": value, "img_name": str(value), "embedding": [1.0, 0.0]}
                for value in (1, 2)
            ]

        def link_verified_identities(self, **kwargs):
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("Background link was not released")
            assert self.closed is False
            self.linked = kwargs["clusters"]
            completed.set()
            return 1

        def close(self):
            self.closed = True

    client = GraphClient()
    monkeypatch.setattr(datastore, "__connect_database", lambda **kwargs: client)
    monkeypatch.setattr(
        datastore,
        "represent",
        lambda **kwargs: [{"embedding": [1.0, 0.0]}],
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(datastore, "_LINK_EXECUTOR", executor)
        try:
            results = datastore.search(
                img="synthetic-image",
                model_name="Facenet",
                detector_backend="skip",
                database_type="neo4j",
                connection=object() if borrowed else None,
                return_type="dict",
            )
            assert [row["id"] for row in results[0]] == [1, 2]
            assert started.wait(timeout=5)
            assert completed.is_set() is False
            assert client.closed is False
        finally:
            release.set()
        assert completed.wait(timeout=5)
    assert client.linked == [[1, 2]]
    assert client.closed is False
