"""Measure real MongoDB identity allocation and preserve concurrent registration IDs."""

import json
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import numpy as np
import pytest

from deepface.modules.database.mongo import MongoDbClient
from deepface.modules.exceptions import DuplicateEntryError

pymongo = pytest.importorskip("pymongo")

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)


class CounterCommands(pymongo.monitoring.CommandListener):
    def __init__(self):
        self.increments = []
        self.lock = Lock()

    def started(self, event):
        if (
            event.command_name == "findAndModify"
            and event.command.get("findAndModify") == "counters"
        ):
            with self.lock:
                self.increments.append(event.command["update"]["$inc"]["seq"])

    def succeeded(self, event):
        pass

    def failed(self, event):
        pass


@pytest.fixture
def database():
    uri = os.environ.get("DEEPFACE_MONGO_URI")
    if not uri:
        pytest.skip("set DEEPFACE_MONGO_URI to an isolated test server")
    observer = CounterCommands()
    name = "deepface_" + uuid.uuid4().hex
    connection = pymongo.MongoClient(
        uri, serverSelectionTimeoutMS=10000, event_listeners=[observer]
    )
    client = MongoDbClient(connection=connection, db_name=name)
    try:
        yield client, observer
    finally:
        connection.drop_database(name)
        connection.close()


def records(count, offset=0):
    return [
        dict(
            img_name=f"image-{index}.png",
            face=np.full((2, 2, 3), index % 256, dtype=np.uint8),
            embedding=[float(index + 1), 0.25],
            **CRITERIA,
        )
        for index in range(offset, offset + count)
    ]


@pytest.mark.parametrize("count", [1, 16, 256])
@pytest.mark.parametrize("batch_size", [1, 100])
def test_registration_reserves_one_identity_block(database, count, batch_size):
    client, observer = database
    durations, calls = [], []
    for repeat in range(3):
        batch = records(count, offset=repeat * count)
        previous_calls = len(observer.increments)
        start = time.perf_counter()
        assert client.insert_embeddings(batch, batch_size=batch_size) == count
        durations.append(1000 * (time.perf_counter() - start))
        calls.append(len(observer.increments) - previous_calls)

    rows = client.fetch_all_embeddings(**CRITERIA)
    assert [row["id"] for row in rows] == list(range(1, 3 * count + 1))
    assert [row["img_name"] for row in rows] == [f"image-{i}.png" for i in range(3 * count)]
    assert [row["embedding"] for row in rows] == [[float(i + 1), 0.25] for i in range(3 * count)]
    assert client.counters.find_one({"_id": "embedding_id"})["seq"] == 3 * count
    print(
        json.dumps(
            dict(
                phase=os.environ.get("DEEPFACE_BENCHMARK_PHASE", "test"),
                count=count,
                batch_size=batch_size,
                median_ms=statistics.median(durations),
                counter_calls=calls,
            )
        )
    )
    assert calls == [1, 1, 1]
    assert observer.increments == [count, count, count]


@pytest.mark.parametrize("workers", [2, 4])
def test_concurrent_blocks_never_reuse_an_identity(database, workers):
    client, observer = database
    start = Barrier(workers)
    count = 32

    def register(worker):
        batch = records(count, offset=worker * count)
        start.wait(timeout=10)
        return client.insert_embeddings(batch, batch_size=7)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        assert list(executor.map(register, range(workers))) == [count] * workers
    rows = client.fetch_all_embeddings(**CRITERIA)
    assert [row["id"] for row in rows] == list(range(1, workers * count + 1))
    assert len({row["img_name"] for row in rows}) == workers * count
    assert client.counters.find_one({"_id": "embedding_id"})["seq"] == workers * count
    assert observer.increments == [count] * workers


@pytest.mark.parametrize("multiple", [False, True])
def test_duplicate_registration_preserves_results_and_allocated_gaps(database, multiple):
    client, _ = database
    assert client.insert_embeddings(records(1)) == 1
    if multiple:
        with pytest.raises(DuplicateEntryError):
            client.insert_embeddings(records(2))
        expected_ids = [1, 3]
        expected_counter = 3
    else:
        assert client.insert_embeddings(records(1)) == 0
        expected_ids = [1]
        expected_counter = 2
    assert [row["id"] for row in client.fetch_all_embeddings(**CRITERIA)] == expected_ids
    assert client.counters.find_one({"_id": "embedding_id"})["seq"] == expected_counter
