"""Concurrent MongoDB client initialization must preserve the shared identity counter."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from deepface.modules.database.mongo import MongoDbClient

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)


@pytest.fixture
def database():
    pymongo = pytest.importorskip("pymongo")
    uri = os.environ.get("DEEPFACE_MONGO_URI")
    if not uri:
        pytest.skip("set DEEPFACE_MONGO_URI to an isolated test server")
    name = "deepface_" + uuid.uuid4().hex
    administrator = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10000)
    administrator.admin.command("ping")
    bootstrap = MongoDbClient(connection=administrator, db_name=name)
    connections = []

    def connected_clients(count, independent):
        if independent:
            for _ in range(count):
                connection = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10000)
                connection.admin.command("ping")
                connections.append(connection)
            return connections
        return [administrator] * count

    try:
        yield bootstrap, connected_clients, name
    finally:
        for connection in connections:
            connection.close()
        administrator.drop_database(name)
        administrator.close()


def initialize_together(connections, name):
    start = Barrier(len(connections))

    def initialize(connection):
        start.wait(timeout=10)
        return MongoDbClient(connection=connection, db_name=name)

    clients, errors = [], []
    with ThreadPoolExecutor(max_workers=len(connections)) as pool:
        futures = [pool.submit(initialize, connection) for connection in connections]
        for future in futures:
            try:
                clients.append(future.result(timeout=20))
            except Exception as error:
                errors.append(error)
    assert not errors, [repr(error) for error in errors]
    return clients


def register_together(clients, first_id):
    def insert(index):
        record = dict(
            img_name=f"person-{index}.png",
            face=np.full((2, 2, 3), index, dtype=np.uint8),
            embedding=[float(index + 1), 0.25],
            **CRITERIA,
        )
        return clients[index].insert_embeddings([record])

    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        assert list(pool.map(insert, range(len(clients)))) == [1] * len(clients)
    rows = clients[0].fetch_all_embeddings(**CRITERIA)
    assert [row["id"] for row in rows] == list(range(first_id, first_id + len(clients)))
    assert {row["img_name"] for row in rows} == {
        f"person-{index}.png" for index in range(len(clients))
    }


@pytest.mark.parametrize("independent", [False, True], ids=["shared", "independent"])
@pytest.mark.parametrize("workers", [2, 4])
def test_first_concurrent_clients_initialize_one_counter(
    database, monkeypatch, independent, workers
):
    bootstrap, connected_clients, name = database
    bootstrap.counters.delete_many({})
    connections = connected_clients(workers, independent)
    collection_type = type(bootstrap.counters)
    original_find_one = collection_type.find_one
    reads_finished = Barrier(workers)

    def simultaneous_empty_reads(self, *args, **kwargs):
        document = original_find_one(self, *args, **kwargs)
        if self.full_name == bootstrap.counters.full_name and document is None:
            # Use the real read result, then align the original read/insert race window.
            reads_finished.wait(timeout=10)
        return document

    monkeypatch.setattr(collection_type, "find_one", simultaneous_empty_reads)

    clients = initialize_together(connections, name)

    assert bootstrap.counters.count_documents({}) == 1
    assert bootstrap.counters.find_one({"_id": "embedding_id"})["seq"] == 0
    register_together(clients, first_id=1)


@pytest.mark.parametrize("independent", [False, True], ids=["shared", "independent"])
def test_reinitialization_never_resets_allocated_identities(database, independent):
    bootstrap, connected_clients, name = database
    bootstrap.counters.update_one({"_id": "embedding_id"}, {"$set": {"seq": 37}})
    connections = connected_clients(4, independent)

    clients = initialize_together(connections, name)

    assert bootstrap.counters.find_one({"_id": "embedding_id"})["seq"] == 37
    register_together(clients, first_id=38)
