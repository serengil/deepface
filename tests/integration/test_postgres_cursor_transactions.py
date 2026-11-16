"""PostgreSQL streaming reads have a bounded transaction without committing caller work."""

import os
import uuid

import numpy as np
import pytest

from deepface.modules import datastore
from deepface.modules.database.postgres import PostgresClient

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)


@pytest.fixture(params=[False, True], ids=["transactional", "autocommit"])
def database(request):
    psycopg = pytest.importorskip("psycopg")
    uri = os.environ.get("DEEPFACE_POSTGRES_URI")
    if not uri:
        pytest.skip("set DEEPFACE_POSTGRES_URI to an isolated test database")

    schema = "deepface_" + uuid.uuid4().hex
    administrator = psycopg.connect(uri, autocommit=True)
    administrator.execute(
        psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema))
    )
    options = "-c search_path=" + schema
    connection = psycopg.connect(uri, autocommit=request.param, options=options)
    observer = psycopg.connect(uri, autocommit=True, options=options)
    client = PostgresClient(connection=connection)
    records = [
        dict(
            img_name=f"person-{i}.png",
            face=np.full((2, 2, 3), i, dtype=np.uint8),
            embedding=[float(i + 1), 0.25],
            **CRITERIA,
        )
        for i in range(5)
    ]
    assert client.insert_embeddings(records, batch_size=2) == 5
    try:
        yield client, observer, psycopg
    finally:
        connection.close()
        observer.close()
        administrator.execute(
            psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema))
        )
        administrator.close()


def test_batched_read_finishes_its_own_transaction(database):
    client, _observer, psycopg = database

    rows = client.fetch_all_embeddings(**CRITERIA, batch_size=2)

    assert [row["img_name"] for row in rows] == [f"person-{i}.png" for i in range(5)]
    assert [row["embedding"] for row in rows] == [[float(i + 1), 0.25] for i in range(5)]
    assert all(all(row[key] == value for key, value in CRITERIA.items()) for row in rows)
    assert client.conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


def test_empty_read_finishes_its_own_transaction(database):
    client, _observer, psycopg = database

    rows = client.fetch_all_embeddings(**dict(CRITERIA, model_name="Facenet512"), batch_size=2)

    assert rows == []
    assert client.conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


def test_streaming_read_does_not_commit_an_existing_transaction(database):
    client, observer, psycopg = database
    client.conn.execute("CREATE TABLE caller_work (value INTEGER)")
    client.conn.commit()

    with client.conn.transaction():
        client.conn.execute("INSERT INTO caller_work VALUES (1)")
        assert len(client.fetch_all_embeddings(**CRITERIA, batch_size=2)) == 5
        assert client.conn.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
        assert observer.execute("SELECT count(*) FROM caller_work").fetchone()[0] == 0
        raise psycopg.Rollback

    assert observer.execute("SELECT count(*) FROM caller_work").fetchone()[0] == 0
    assert client.conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


def test_public_exact_search_accepts_both_connection_modes(database, monkeypatch):
    client, _observer, _psycopg = database
    original_mode = client.conn.autocommit
    monkeypatch.setattr(datastore, "represent", lambda **_kwargs: [{"embedding": [1.0, 0.25]}])

    rows = datastore.search(
        img="fixed-query",
        model_name="Facenet",
        detector_backend="skip",
        database_type="postgres",
        connection=client.conn,
        search_method="exact",
        distance_metric="euclidean",
        similarity_search=True,
        return_type="dict",
    )[0]

    assert [row["img_name"] for row in rows] == [f"person-{i}.png" for i in range(5)]
    assert [row["distance"] for row in rows] == [0, 1, 2, 3, 4]
    assert not client.conn.closed
    assert client.conn.autocommit is original_mode
