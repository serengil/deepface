"""Only internally created registration clients are closed, including on errors."""

from copy import deepcopy
import numpy as np
import pytest
from deepface.modules import datastore


class RegistrationError(RuntimeError):
    pass


class MemoryDatabase:
    def __init__(self):
        self.closed = 0
        self.inserted = []
        self.fail = None

    def close(self):
        self.closed += 1

    def insert_embeddings(self, rows, batch_size):
        assert batch_size == 100
        if self.fail is not None:
            raise self.fail
        self.inserted = deepcopy(rows)
        return len(rows)


@pytest.fixture
def storage(monkeypatch):
    db = MemoryDatabase()
    monkeypatch.setattr(datastore, "__connect_database", lambda **kwargs: db)
    monkeypatch.setitem(
        datastore.database_inventory, "postgres", {"is_graph_db": False}
    )
    monkeypatch.setitem(datastore.database_inventory, "neo4j", {"is_graph_db": True})

    def represent(**kwargs):
        return [
            {
                "embedding": [1.0, 0.0],
                "face": np.zeros((2, 2, 3)),
                "facial_area": {"x": 0, "y": 0, "w": 2, "h": 2},
            }
        ]

    monkeypatch.setattr(datastore, "represent", represent)
    monkeypatch.setattr(
        datastore.image_utils,
        "load_image",
        lambda source: (np.zeros((2, 2, 3)), source),
    )
    return db


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("stage", ["represent", "insert", "image_load", "attributes"])
def test_cleanup_for_failures(storage, monkeypatch, owned, stage):
    error = RegistrationError(stage)

    def fail(*args, **kwargs):
        raise error

    database = "postgres"
    if stage == "represent":
        monkeypatch.setattr(datastore, "represent", fail)
    elif stage == "insert":
        storage.fail = error
    elif stage == "image_load":
        database = "neo4j"
        monkeypatch.setattr(datastore.image_utils, "load_image", fail)
    else:
        database = "neo4j"
        monkeypatch.setattr(datastore, "__assign_attributes", fail)
    with pytest.raises(RegistrationError) as caught:
        datastore.register(
            "synthetic.png",
            model_name="Facenet512",
            detector_backend="skip",
            database_type=database,
            connection=None if owned else object(),
        )
    assert caught.value is error
    assert storage.closed == int(owned)


@pytest.mark.parametrize("owned", [False, True])
def test_success_preserves_payload_and_connection_ownership(storage, owned):
    result = datastore.register(
        "synthetic.png",
        img_name="sample",
        model_name="Facenet512",
        detector_backend="skip",
        align=False,
        l2_normalize=True,
        connection=None if owned else object(),
    )
    assert result == {"inserted": 1}
    assert storage.closed == int(owned)
    assert storage.inserted[0]["img_name"] == "sample"
    assert storage.inserted[0]["model_name"] == "Facenet512"
    assert storage.inserted[0]["embedding"] == [1.0, 0.0]
    assert storage.inserted[0]["aligned"] is False
    assert storage.inserted[0]["l2_normalized"] is True


def test_connection_creation_failure_is_not_replaced(storage, monkeypatch):
    error = RegistrationError("connect")

    def connect(**kwargs):
        raise error

    monkeypatch.setattr(datastore, "__connect_database", connect)
    with pytest.raises(RegistrationError) as caught:
        datastore.register("synthetic.png")
    assert caught.value is error
    assert storage.closed == 0


def test_repeated_failures_release_every_owned_client(monkeypatch, storage):
    clients = []

    def connect(**kwargs):
        db = MemoryDatabase()
        db.fail = RegistrationError("insert")
        clients.append(db)
        return db

    monkeypatch.setattr(datastore, "__connect_database", connect)
    for _ in range(20):
        with pytest.raises(RegistrationError):
            datastore.register("synthetic.png", detector_backend="skip")
    assert len(clients) == 20
    assert all(client.closed == 1 for client in clients)
