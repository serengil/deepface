import sqlite3

import numpy as np
import pytest

from deepface.modules.database.sqlite import SqliteClient


@pytest.fixture
def gallery():
    conn = sqlite3.connect(":memory:")
    client = SqliteClient(connection=conn)
    rows = [
        dict(
            img_name=f"{i}.png",
            face=np.zeros((1, 1)),
            embedding=[float(i)],
            model_name="test",
            detector_backend="skip",
            aligned=True,
            l2_normalized=False,
        )
        for i in range(3)
    ]
    client.insert_embeddings(rows)
    yield conn, client, rows
    conn.close()


@pytest.mark.parametrize("size", [-1, 0])
def test_nonpositive_batch_sizes_are_rejected(gallery, size):
    conn, client, rows = gallery
    with pytest.raises(ValueError, match="batch_size"):
        client.insert_embeddings(rows, batch_size=size)
    with pytest.raises(ValueError, match="batch_size"):
        client.fetch_all_embeddings("test", "skip", True, False, batch_size=size)
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone() == (3,)


def test_chunking_preserves_unique_id_order(gallery):
    _, client, _ = gallery
    ids = [3] + list(range(1000, 1500)) + [1, 3, 2]
    assert [r["id"] for r in client.search_by_id(ids)] == [1, 2, 3]


def test_wrapping_connection_preserves_pending_caller_transaction(gallery):
    conn, _, _ = gallery
    conn.execute("UPDATE embeddings SET img_name = 'temporary' WHERE id = 1")
    SqliteClient(connection=conn)
    conn.rollback()
    assert conn.execute("SELECT img_name FROM embeddings WHERE id = 1").fetchone() == ("0.png",)
