"""Run with pymilvus and milvus-lite; no model weights are needed."""

from types import SimpleNamespace

import pytest

sdk = pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")

from deepface.modules.database import milvus


def test_milvus_lite_fetches_more_than_query_limit(tmp_path, monkeypatch):
    client = sdk.MilvusClient(uri=str(tmp_path / "gallery.db"))
    adapter = milvus.MilvusClient(connection=client)
    monkeypatch.setattr(
        milvus, "build_model", lambda **kw: SimpleNamespace(output_shape=2)
    )
    criteria = dict(
        model_name="Facenet512", detector_backend="skip", aligned=True, l2_normalized=False
    )
    try:
        adapter.initialize_database(**criteria)
        collection = client.list_collections()[0]
        count = 16385
        for start in range(0, count, 1000):
            client.insert(
                collection_name=collection,
                data=[
                    {"id": i, "embedding": [float(i), 0.0], "img_name": f"image-{i}"}
                    for i in range(start, min(start + 1000, count))
                ],
            )
        client.flush(collection_name=collection)

        records = adapter.fetch_all_embeddings(**criteria, batch_size=1000)

        assert len(records) == count
        assert {row["id"] for row in records} == set(range(count))
        last = next(row for row in records if row["id"] == count - 1)
        assert last["img_name"] == f"image-{count - 1}"
        assert last["embedding"] == [float(count - 1), 0.0]
    finally:
        client.close()
