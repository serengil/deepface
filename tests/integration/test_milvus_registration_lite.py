"""Stable Milvus identities remain unique when faces are registered again."""

from types import SimpleNamespace

import numpy as np
import pytest

sdk = pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")

from deepface.modules.database import milvus


CRITERIA = dict(model_name="Facenet512", detector_backend="skip", aligned=True, l2_normalized=False)


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    client = sdk.MilvusClient(uri=str(tmp_path / "registration.db"))
    adapter = milvus.MilvusClient(connection=client)
    monkeypatch.setattr(milvus, "build_model", lambda **kwargs: SimpleNamespace(output_shape=2))
    try:
        yield adapter, client
    finally:
        client.close()


def record(value, name):
    return {
        **CRITERIA,
        "face": np.zeros((2, 2, 3), dtype=np.float32),
        "embedding": [value, 0.5],
        "img_name": name,
    }


def count_entities(client):
    collection = client.list_collections()[0]
    client.flush(collection_name=collection)
    return client.query(collection_name=collection, filter="", output_fields=["count(*)"])[0][
        "count(*)"
    ]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_repeated_registration_replaces_existing_entities(gallery, batch_size):
    adapter, client = gallery
    assert adapter.insert_embeddings([record(0.25, "old.jpg"), record(0.75, "other.jpg")]) == 2
    before = adapter.fetch_all_embeddings(**CRITERIA)

    for _ in range(3):
        assert (
            adapter.insert_embeddings(
                [record(0.25, "latest.jpg"), record(0.75, "other.jpg")], batch_size=batch_size
            )
            == 2
        )

    assert count_entities(client) == 2
    after = adapter.fetch_all_embeddings(**CRITERIA)
    assert {row["id"] for row in after} == {row["id"] for row in before}
    assert {row["img_name"] for row in after} == {"latest.jpg", "other.jpg"}
    for row in after:
        fetched = adapter.fetch_embedding(row["id"], **CRITERIA)
        assert fetched["img_name"] == row["img_name"]
        assert fetched["embedding"] == pytest.approx(row["embedding"])


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_repeated_identity_within_input_keeps_last_record(gallery, batch_size):
    adapter, client = gallery
    records = [
        record(0.25, "first.jpg"),
        record(0.25, "middle.jpg"),
        record(0.75, "other.jpg"),
        record(0.25, "latest.jpg"),
    ]

    # The return value continues to count the supplied embeddings, including updates.
    assert adapter.insert_embeddings(records, batch_size=batch_size) == 4

    assert count_entities(client) == 2
    stored = adapter.fetch_all_embeddings(**CRITERIA)
    assert len(stored) == 2
    assert {row["img_name"] for row in stored} == {"latest.jpg", "other.jpg"}


def test_distinct_embeddings_remain_distinct(gallery):
    adapter, client = gallery
    assert adapter.insert_embeddings([record(0.25, "first.jpg"), record(0.75, "second.jpg")]) == 2
    assert count_entities(client) == 2
    stored = adapter.fetch_all_embeddings(**CRITERIA)
    assert len({row["id"] for row in stored}) == 2
    assert {tuple(row["embedding"]) for row in stored} == {(0.25, 0.5), (0.75, 0.5)}


def test_empty_registration_does_not_create_collection(gallery):
    adapter, client = gallery
    with pytest.raises(ValueError, match="No embeddings"):
        adapter.insert_embeddings([])
    assert client.list_collections() == []
