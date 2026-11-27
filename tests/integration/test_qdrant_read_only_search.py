# built-in dependencies
from types import SimpleNamespace
from unittest.mock import patch

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules.database.qdrant import QdrantClient

qdrant_client = pytest.importorskip("qdrant_client")


@pytest.fixture(params=[":memory:", "disk"])
def connection(request, tmp_path):
    if request.param == "disk":
        client = qdrant_client.QdrantClient(path=str(tmp_path / "gallery"))
    else:
        client = qdrant_client.QdrantClient(location=":memory:")
    try:
        yield client
    finally:
        client.close()


@pytest.mark.parametrize("normalized", [False, True])
def test_missing_gallery_search_leaves_collection_catalog_unchanged(
    connection, normalized
):
    client = QdrantClient(connection=connection)
    with patch(
        "deepface.modules.database.qdrant.build_model",
        return_value=SimpleNamespace(output_shape=3),
    ) as build_model:
        assert client.search_by_vector([1.0, 0.0, 0.0], l2_normalized=normalized) == []

    assert connection.get_collections().collections == []
    build_model.assert_not_called()
    assert client.fetch_all_embeddings("VGG-Face", "opencv", True, normalized) == []


@pytest.mark.parametrize("normalized", [False, True])
def test_search_still_reads_registered_vectors_without_loading_a_model(
    connection, normalized
):
    client = QdrantClient(connection=connection)
    records = [
        {
            "img_name": name,
            "face": np.full((2, 2, 3), index, dtype=np.float32),
            "model_name": "VGG-Face",
            "detector_backend": "opencv",
            "aligned": True,
            "l2_normalized": normalized,
            "embedding": embedding,
        }
        for index, (name, embedding) in enumerate(
            [("same.jpg", [1.0, 0.0, 0.0]), ("other.jpg", [0.0, 1.0, 0.0])]
        )
    ]
    with patch(
        "deepface.modules.database.qdrant.build_model",
        return_value=SimpleNamespace(output_shape=3),
    ):
        assert client.insert_embeddings(records) == 2

    with patch("deepface.modules.database.qdrant.build_model") as build_model:
        results = client.search_by_vector(
            [1.0, 0.0, 0.0], l2_normalized=normalized, limit=2
        )
        assert [result["img_name"] for result in results] == ["same.jpg", "other.jpg"]
        assert results[0]["distance"] == pytest.approx(0.0)
        assert results[1]["distance"] == pytest.approx(1.0 if normalized else 2**0.5)
        assert (
            len(
                client.search_by_vector(
                    [1.0, 0.0, 0.0], l2_normalized=normalized, limit=1
                )
            )
            == 1
        )
    build_model.assert_not_called()


def test_existing_empty_collection_stays_readable(connection):
    client = QdrantClient(connection=connection)
    with patch(
        "deepface.modules.database.qdrant.build_model",
        return_value=SimpleNamespace(output_shape=3),
    ):
        client.initialize_database()
    before = connection.get_collections()

    with patch("deepface.modules.database.qdrant.build_model") as build_model:
        assert client.search_by_vector([1.0, 0.0, 0.0]) == []

    assert connection.get_collections() == before
    build_model.assert_not_called()
