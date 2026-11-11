"""Search must release its local database even while an error traceback is retained."""

import numpy as np
import pytest

from deepface.modules import datastore

qdrant = pytest.importorskip("qdrant_client")

COLLECTION = "embeddings_facenet_skip_aligned_raw"


@pytest.fixture
def local_database(tmp_path, monkeypatch):
    path = str(tmp_path / "gallery")
    client = qdrant.QdrantClient(path=path)
    try:
        client.create_collection(
            COLLECTION,
            vectors_config=qdrant.models.VectorParams(
                size=2, distance=qdrant.models.Distance.EUCLID
            ),
        )
        client.upsert(
            COLLECTION,
            points=[
                qdrant.models.PointStruct(
                    id=1, vector=[1.0, 0.0], payload={"img_name": "known-face"}
                )
            ],
        )
    finally:
        client.close()

    monkeypatch.setattr(
        datastore,
        "represent",
        lambda **kwargs: [
            {
                "embedding": [1.0, 0.0],
                "facial_area": {"x": 0, "y": 0, "w": 2, "h": 2},
            }
        ],
    )
    return path, {
        "img": np.ones((2, 2, 3), dtype=np.uint8),
        "model_name": "Facenet",
        "detector_backend": "skip",
        "database_type": "qdrant",
        "connection_details": {"path": path},
        "return_type": "dict",
    }


def assert_reopens(path, expected_count=1):
    reopened = qdrant.QdrantClient(path=path)
    try:
        assert reopened.count(COLLECTION).count == expected_count
    finally:
        reopened.close()


@pytest.mark.parametrize("search_method", ["exact", "ann"])
def test_successful_owned_search_releases_local_storage(local_database, search_method):
    path, options = local_database
    result = datastore.search(**options, search_method=search_method)
    assert result[0][0]["img_name"] == "known-face"
    assert result[0][0]["distance"] == pytest.approx(0)
    assert_reopens(path)


@pytest.mark.parametrize("search_method", ["exact", "ann"])
def test_empty_database_search_releases_local_storage(local_database, search_method):
    path, options = local_database
    client = qdrant.QdrantClient(path=path)
    try:
        client.delete(COLLECTION, points_selector=[1])
    finally:
        client.close()

    with pytest.raises(ValueError, match="No embeddings found") as caught:
        datastore.search(**options, search_method=search_method)
    assert caught.value.__traceback__ is not None
    assert_reopens(path, expected_count=0)


@pytest.mark.parametrize("search_method", ["exact", "ann"])
@pytest.mark.parametrize("stage", ["representation", "query", "dataframe"])
def test_failed_owned_search_releases_local_storage(
    local_database, monkeypatch, search_method, stage
):
    path, options = local_database
    error = RuntimeError(f"failure during {stage}")

    def fail(*args, **kwargs):
        raise error

    if stage == "representation":
        monkeypatch.setattr(datastore, "represent", fail)
    elif stage == "query":
        query_method = "scroll" if search_method == "exact" else "query_points"
        monkeypatch.setattr(qdrant.QdrantClient, query_method, fail)
    else:
        pandas = pytest.importorskip("pandas")
        options["return_type"] = "pandas"
        monkeypatch.setattr(pandas, "DataFrame", fail)

    with pytest.raises(RuntimeError) as caught:
        datastore.search(**options, search_method=search_method)
    assert caught.value is error
    # Retaining the traceback is normal for callers that log or inspect failures.
    # The old search leaves its client in that traceback, still holding this lock.
    assert caught.value.__traceback__ is not None
    assert_reopens(path)


@pytest.mark.parametrize("fails", [False, True])
def test_borrowed_local_client_remains_usable(local_database, monkeypatch, fails):
    path, options = local_database
    borrowed = qdrant.QdrantClient(path=path)
    options["connection"] = borrowed
    try:
        if fails:
            def fail(**kwargs):
                raise RuntimeError("representation failed")

            monkeypatch.setattr(datastore, "represent", fail)
            with pytest.raises(RuntimeError, match="representation failed"):
                datastore.search(**options)
        else:
            assert datastore.search(**options)[0][0]["img_name"] == "known-face"
        assert borrowed.count(COLLECTION).count == 1
    finally:
        borrowed.close()
    assert_reopens(path)
