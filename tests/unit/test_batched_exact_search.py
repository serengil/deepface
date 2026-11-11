"""Exact database searches preserve scalar metric results and row metadata."""

from unittest.mock import Mock

import numpy as np
import pytest

from deepface.modules import datastore, verification


def search_fixture(monkeypatch, vectors, queries):
    records = [
        dict(id=i, img_name=f"image-{i}", embedding=row.tolist())
        for i, row in enumerate(vectors)
    ]
    db = Mock()
    db.fetch_all_embeddings.return_value = records
    monkeypatch.setattr(datastore, "__connect_database", lambda **kwargs: db)
    monkeypatch.setattr(
        datastore,
        "__get_embeddings",
        lambda **kwargs: [
            dict(embedding=row.tolist(), facial_area=dict(x=i, y=0, w=1, h=1))
            for i, row in enumerate(queries)
        ],
    )
    return records


@pytest.mark.parametrize("metric", ["cosine", "angular", "euclidean", "euclidean_l2"])
@pytest.mark.parametrize("limit", [None, 3])
def test_matches_scalar_distances_order_and_metadata(monkeypatch, metric, limit):
    rng = np.random.default_rng(27)
    vectors, queries = rng.normal(size=(600, 8)), rng.normal(size=(3, 8))
    records = search_fixture(monkeypatch, vectors, queries)
    actual = datastore.search(
        img="synthetic",
        database_type="sqlite",
        distance_metric=metric,
        return_type="dict",
        similarity_search=True,
        k=limit,
        connection=object(),
    )
    for query_index, query in enumerate(queries):
        if metric == "cosine":
            expected = [
                verification.find_cosine_distance(row, query) for row in vectors
            ]
        elif metric == "angular":
            expected = [
                verification.find_angular_distance(row, query) for row in vectors
            ]
        else:
            expected = [
                verification.find_euclidean_distance(
                    verification.l2_normalize(row) if metric == "euclidean_l2" else row,
                    (
                        verification.l2_normalize(query)
                        if metric == "euclidean_l2"
                        else query
                    ),
                )
                for row in vectors
            ]
        order = sorted(range(len(records)), key=lambda i: expected[i])[:limit]
        assert [row["id"] for row in actual[query_index]] == order
        assert [row["img_name"] for row in actual[query_index]] == [
            records[i]["img_name"] for i in order
        ]
        assert all(row["target_x"] == query_index for row in actual[query_index])
        np.testing.assert_allclose(
            [row["distance"] for row in actual[query_index]],
            np.array(expected)[order],
            rtol=1e-12,
            atol=1e-12,
        )


def test_small_norm_cosine_keeps_scalar_threshold_semantics(monkeypatch):
    # The public batch distance helper adds epsilon to norms, unlike exact scalar search.
    search_fixture(
        monkeypatch, np.array([[1e-12, 0.0], [-1e-12, 0.0]]), np.array([[1e-12, 0.0]])
    )
    actual = datastore.search(
        img="synthetic", database_type="sqlite", return_type="dict", connection=object()
    )
    assert [row["id"] for row in actual[0]] == [0]
    assert actual[0][0]["distance"] == 0.0
