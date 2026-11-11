"""Deduplication claims must not survive a transaction that never writes records."""

from unittest.mock import patch

import numpy as np
import pytest

fakeredis = pytest.importorskip("fakeredis")

from deepface.modules.database.redis import RedisClient
from deepface.modules.exceptions import DuplicateEntryError


def record(value=1.0):
    return dict(
        img_name="example.png",
        face=np.zeros((2, 2, 3)),
        embedding=[value, 0.25],
        model_name="VGG-Face",
        detector_backend="opencv",
        aligned=True,
        l2_normalized=False,
    )


def failing_transaction(connection):
    original = connection.pipeline

    def pipeline(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("transaction", True):

            def fail_before_exec(*unused, **ignored):
                raise RuntimeError("transaction not sent")

            result.execute = fail_before_exec
        return result

    return patch.object(connection, "pipeline", side_effect=pipeline)


def test_failed_transaction_does_not_leave_a_dedup_claim():
    connection = fakeredis.FakeRedis()
    client = RedisClient(connection=connection)
    with failing_transaction(connection), pytest.raises(RuntimeError, match="not sent"):
        client.insert_embeddings([record()])
    assert connection.keys("deepface:dedup:*") == []
    assert connection.keys("deepface:embedding:*") == []
    assert client.insert_embeddings([record()]) == 1
    assert len(client.fetch_all_embeddings("VGG-Face", "opencv", True, False)) == 1


def test_existing_record_survives_an_unrelated_failed_registration():
    connection = fakeredis.FakeRedis()
    client = RedisClient(connection=connection)
    assert client.insert_embeddings([record()]) == 1
    with failing_transaction(connection), pytest.raises(RuntimeError):
        client.insert_embeddings([record(2.0)])
    assert client.insert_embeddings([record()]) == 0
    assert client.insert_embeddings([record(2.0)]) == 1
    assert len(client.fetch_all_embeddings("VGG-Face", "opencv", True, False)) == 2


def test_duplicate_within_one_batch_writes_no_records():
    connection = fakeredis.FakeRedis()
    client = RedisClient(connection=connection)
    with pytest.raises(DuplicateEntryError):
        client.insert_embeddings([record(), record()])
    assert connection.keys("deepface:dedup:*") == []
    assert client.fetch_all_embeddings("VGG-Face", "opencv", True, False) == []


def test_successful_records_and_repeat_registration_keep_existing_semantics():
    connection = fakeredis.FakeRedis()
    client = RedisClient(connection=connection)
    assert client.insert_embeddings([record(), record(2.0)], batch_size=1) == 2
    rows = client.fetch_all_embeddings("VGG-Face", "opencv", True, False)
    assert [row["embedding"] for row in rows] == [[1.0, 0.25], [2.0, 0.25]]
    assert client.insert_embeddings([record()]) == 0


def test_competing_registrations_store_one_complete_record():
    connection = fakeredis.FakeRedis()
    first, second = RedisClient(connection=connection), RedisClient(
        connection=connection
    )
    original = connection.pipeline
    competing_counts = []

    def pipeline(*args, **kwargs):
        result = original(*args, **kwargs)
        execute = result.execute

        def execute_with_competitor(*execute_args, **execute_kwargs):
            if not competing_counts:
                competing_counts.append(None)
                competing_counts[0] = second.insert_embeddings([record()])
            return execute(*execute_args, **execute_kwargs)

        result.execute = execute_with_competitor
        return result

    with patch.object(connection, "pipeline", side_effect=pipeline):
        first_count = first.insert_embeddings([record()])
    assert first_count + competing_counts[0] == 1
    rows = first.fetch_all_embeddings("VGG-Face", "opencv", True, False)
    assert len(rows) == 1
    (dedup_key,) = connection.keys("deepface:dedup:*")
    assert int(connection.get(dedup_key)) == rows[0]["id"]
