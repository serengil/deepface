"""Milvus scores must use the same units as DeepFace distance thresholds."""

from types import SimpleNamespace

import pytest
import numpy as np

from deepface.modules import datastore
from deepface.modules.database.milvus import MilvusClient


@pytest.mark.parametrize(
    "normalized,score,expected",
    [
        (True, 1.0, 0.0),
        (True, 0.0, 1.0),
        (True, -1.0, 2.0),
        (True, 0.75, 0.25),
        (True, 0.71, 0.29),
        (True, 0.69, 0.31),
        (True, 1.0000001, 0.0),
        (True, -1.0000001, 2.0),
        (False, 0.0, 0.0),
        (False, 100.0, 10.0),
        (False, 0.0625, 0.25),
        (False, 23.55**2, 23.55),
        (False, 23.57**2, 23.57),
        (False, -1e-12, 0.0),
    ],
)
def test_milvus_score_conversion(normalized, score, expected):
    def search(**kwargs):
        assert kwargs["search_params"]["metric_type"] == (
            "COSINE" if normalized else "L2"
        )
        return [[{"id": 7, "distance": score, "entity": {"img_name": "image.png"}}]]

    # Only the SDK boundary is replaced; run the actual adapter method.
    adapter = object.__new__(MilvusClient)
    adapter.client = SimpleNamespace(has_collection=lambda name: True, search=search)
    result = adapter.search_by_vector([1.0, 0.0], l2_normalized=normalized)
    assert result == [
        {"id": 7, "distance": pytest.approx(expected), "img_name": "image.png"}
    ]


def test_empty_milvus_result():
    adapter = object.__new__(MilvusClient)
    adapter.client = SimpleNamespace(
        has_collection=lambda name: True, search=lambda **kw: [[]]
    )
    assert adapter.search_by_vector([1.0, 0.0]) == []


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("similarity_search", [True, False])
def test_datastore_ann_and_exact_use_same_distance_units(
    monkeypatch, normalized, similarity_search
):
    vectors = (
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        if normalized
        else np.array([[1.0, 0.0], [11.0, 0.0], [31.0, 0.0]])
    )
    records = [
        dict(
            id=i,
            img_name=f"image-{i}",
            embedding=vector.tolist(),
            model_name="Facenet512",
            detector_backend="skip",
            aligned=True,
            l2_normalized=normalized,
        )
        for i, vector in enumerate(vectors)
    ]

    def search(**kwargs):
        query = np.asarray(kwargs["data"][0])
        # Reproduce the documented SDK score units at the database boundary.
        scores = vectors @ query if normalized else ((vectors - query) ** 2).sum(axis=1)
        order = np.argsort(-scores if normalized else scores)
        return [
            [
                dict(id=int(i), distance=float(scores[i]), entity=records[i])
                for i in order
            ]
        ]

    adapter = object.__new__(MilvusClient)
    adapter.client = SimpleNamespace(has_collection=lambda name: True, search=search)
    monkeypatch.setattr(adapter, "fetch_all_embeddings", lambda **kw: records)
    monkeypatch.setattr(datastore, "__connect_database", lambda **kw: adapter)
    monkeypatch.setattr(
        datastore,
        "represent",
        lambda **kw: [dict(embedding=[1.0, 0.0], facial_area=dict(x=0, y=0, w=1, h=1))],
    )
    options = dict(
        img="synthetic",
        model_name="Facenet512",
        detector_backend="skip",
        database_type="milvus",
        connection=adapter.client,
        l2_normalize=normalized,
        similarity_search=similarity_search,
        distance_metric="cosine" if normalized else "euclidean",
        k=3,
    )
    ann = datastore.search(**options, search_method="ann")[0]
    exact = datastore.search(**options, search_method="exact")[0]
    assert ann["id"].tolist() == exact["id"].tolist()
    for column in ["distance", "threshold", "confidence"]:
        np.testing.assert_allclose(ann[column], exact[column])
