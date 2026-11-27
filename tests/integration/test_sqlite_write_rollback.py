"""Exercise failed writes against SQLite's native transaction and locking behavior."""

import sqlite3

import numpy as np
import pytest

from deepface.modules.database.sqlite import SqliteClient
from deepface.modules.exceptions import DuplicateEntryError

CRITERIA = {
    "model_name": "Facenet",
    "detector_backend": "skip",
    "aligned": True,
    "l2_normalized": False,
}


def record(name, value):
    return {
        **CRITERIA,
        "img_name": name,
        "face": np.full((2, 2, 3), value, dtype=np.float32),
        "embedding": [float(value), 1.0, 2.0],
    }


@pytest.fixture
def database(tmp_path):
    path = str(tmp_path / "gallery.sqlite")
    client = SqliteClient(connection_details={"database": path, "timeout": 0})
    observer = sqlite3.connect(path, timeout=0)
    try:
        assert observer.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        yield client, observer
    finally:
        observer.close()
        client.close()


def hold_reader(observer):
    observer.execute("BEGIN")
    observer.execute("SELECT id FROM embeddings").fetchall()


def image_names(observer):
    return [row[0] for row in observer.execute("SELECT img_name FROM embeddings ORDER BY id")]


def test_failed_embedding_commit_is_not_published_by_a_later_write(database):
    client, observer = database
    client.insert_embeddings([record("existing.jpg", 1)])
    hold_reader(observer)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        client.insert_embeddings([record("reported-failed.jpg", 2)])

    observer.rollback()
    client.upsert_embeddings_index(**CRITERIA, index_data=b"recovery index")

    assert image_names(observer) == ["existing.jpg"]
    assert client.get_embeddings_index(**CRITERIA) == b"recovery index"
    assert not client.conn.in_transaction


@pytest.mark.parametrize("previous", [None, b"original index"])
def test_failed_index_commit_is_not_published_by_a_later_registration(database, previous):
    client, observer = database
    if previous is not None:
        client.upsert_embeddings_index(**CRITERIA, index_data=previous)
    hold_reader(observer)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        client.upsert_embeddings_index(**CRITERIA, index_data=b"reported-failed index")

    observer.rollback()
    assert client.insert_embeddings([record("recovered.jpg", 1)]) == 1

    if previous is None:
        with pytest.raises(ValueError, match="No Embeddings index found"):
            client.get_embeddings_index(**CRITERIA)
    else:
        assert client.get_embeddings_index(**CRITERIA) == previous
    assert image_names(observer) == ["recovered.jpg"]
    assert not client.conn.in_transaction


def test_operational_error_rolls_back_the_unfinished_insert_batch(database):
    client, observer = database
    client.insert_embeddings([record("existing.jpg", 1)])

    def check_name(name):
        if name == "rejected.jpg":
            raise ValueError("rejected by the database function")
        return 1

    client.conn.create_function("check_name", 1, check_name)
    client.conn.execute(
        "CREATE TRIGGER validate_image BEFORE INSERT ON embeddings "
        "BEGIN SELECT check_name(NEW.img_name); END"
    )

    with pytest.raises(sqlite3.OperationalError, match="user-defined function"):
        client.insert_embeddings(
            [record("unfinished.jpg", 2), record("rejected.jpg", 3)], batch_size=2
        )

    client.conn.execute("DROP TRIGGER validate_image")
    client.upsert_embeddings_index(**CRITERIA, index_data=b"recovery index")
    assert image_names(observer) == ["existing.jpg"]


def test_successful_batches_and_index_replacement_are_committed(database):
    client, observer = database
    assert (
        client.insert_embeddings(
            [record("one.jpg", 1), record("two.jpg", 2), record("three.jpg", 3)], batch_size=2
        )
        == 3
    )
    client.upsert_embeddings_index(**CRITERIA, index_data=b"first index")
    client.upsert_embeddings_index(**CRITERIA, index_data=b"replacement index")

    assert image_names(observer) == ["one.jpg", "two.jpg", "three.jpg"]
    assert observer.execute("SELECT index_data FROM embeddings_index").fetchall() == [
        (b"replacement index",)
    ]
    assert not client.conn.in_transaction


def test_single_duplicate_keeps_its_existing_result(database):
    client, observer = database
    original = record("existing.jpg", 1)
    assert client.insert_embeddings([original]) == 1
    assert client.insert_embeddings([original]) == 0
    assert image_names(observer) == ["existing.jpg"]
    assert not client.conn.in_transaction


def test_duplicate_rolls_back_earlier_rows_in_the_same_batch(database):
    client, observer = database
    original = record("existing.jpg", 1)
    client.insert_embeddings([original])
    with pytest.raises(DuplicateEntryError):
        client.insert_embeddings([record("unfinished.jpg", 2), original], batch_size=2)
    assert image_names(observer) == ["existing.jpg"]
    assert not client.conn.in_transaction


def test_nonduplicate_integrity_errors_keep_their_original_exception(database):
    client, observer = database
    client.insert_embeddings([record("existing.jpg", 1)])
    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
        client.insert_embeddings([record("unfinished.jpg", 2), record(None, 3)], batch_size=2)
    assert image_names(observer) == ["existing.jpg"]
    assert not client.conn.in_transaction
