"""Exercise registration ownership with an actual on-disk Qdrant local client."""

from types import SimpleNamespace

import numpy as np
import pytest

from deepface.modules import datastore
from deepface.modules.database import qdrant


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_local_client_ownership(tmp_path, monkeypatch, owned, fail):
    sdk = pytest.importorskip("qdrant_client")
    path = str(tmp_path / "database")
    clients = []

    class TrackedClient(qdrant.QdrantClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            clients.append(self._client)

    monkeypatch.setitem(
        datastore.database_inventory,
        "qdrant",
        {**datastore.database_inventory["qdrant"], "client": TrackedClient},
    )
    monkeypatch.setattr(
        qdrant, "build_model", lambda **kwargs: SimpleNamespace(output_shape=2)
    )

    def represent(**kwargs):
        if fail:
            raise ValueError("synthetic inference failure")
        return [
            {
                "embedding": [1.0, 0.0],
                "face": np.zeros((2, 2, 3)),
                "facial_area": {"x": 0, "y": 0, "w": 2, "h": 2},
            }
        ]

    monkeypatch.setattr(datastore, "represent", represent)
    external = None if owned else sdk.QdrantClient(path=path)
    try:
        options = dict(
            img=np.zeros((2, 2, 3)),
            model_name="Facenet512",
            detector_backend="skip",
            database_type="qdrant",
            connection=external,
            connection_details={"path": path},
        )
        if fail:
            with pytest.raises(ValueError, match="synthetic inference failure"):
                datastore.register(**options)
        else:
            assert datastore.register(**options) == {"inserted": 1}

        if owned:
            with pytest.raises(RuntimeError, match="closed"):
                clients[0].get_collections()
            # Opening the same path also proves its exclusive file lock was released.
            reopened = sdk.QdrantClient(path=path)
            try:
                assert len(reopened.get_collections().collections) == (0 if fail else 1)
            finally:
                reopened.close()
        else:
            assert len(external.get_collections().collections) == (0 if fail else 1)
    finally:
        for client in clients:
            client.close()
        if external is not None:
            external.close()
