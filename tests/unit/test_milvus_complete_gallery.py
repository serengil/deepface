"""Exact search must not truncate Milvus galleries at the query result limit."""

from unittest.mock import MagicMock

import pytest

sdk = pytest.importorskip("pymilvus")

from deepface.modules.database.milvus import MilvusClient


@pytest.mark.parametrize("count", [0, 1, 16385])
@pytest.mark.parametrize("batch_size", [1000, 4096])
def test_fetch_all_embeddings_reads_every_page(count, batch_size):
    records = [
        {"id": i, "embedding": [float(i), 0.0], "img_name": f"image-{i}"}
        for i in range(count)
    ]
    client = MagicMock(spec=sdk.MilvusClient)
    client.has_collection.return_value = True
    client.query.return_value = records[:16384]
    iterator = client.query_iterator.return_value
    iterator.next.side_effect = [
        records[start : start + batch_size] for start in range(0, count, batch_size)
    ] + [[]]
    adapter = MilvusClient(connection=client)

    result = adapter.fetch_all_embeddings("Facenet512", "skip", True, False, batch_size)

    assert result == records


def test_fetch_all_embeddings_closes_iterator_on_query_error():
    client = MagicMock(spec=sdk.MilvusClient)
    client.has_collection.return_value = True
    iterator = client.query_iterator.return_value
    iterator.next.side_effect = RuntimeError("query interrupted")
    adapter = MilvusClient(connection=client)

    with pytest.raises(RuntimeError, match="query interrupted"):
        adapter.fetch_all_embeddings("Facenet512", "skip", True, False)

    iterator.close.assert_called_once_with()


def test_fetch_all_embeddings_without_registered_collection():
    client = MagicMock(spec=sdk.MilvusClient)
    client.has_collection.return_value = False
    adapter = MilvusClient(connection=client)

    assert adapter.fetch_all_embeddings("Facenet512", "skip", True, False) == []
    client.query_iterator.assert_not_called()
