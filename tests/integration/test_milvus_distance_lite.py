"""Run with pip install 'pymilvus[milvus-lite]'; no model weights are needed."""

from types import SimpleNamespace

import numpy as np
import pytest

from deepface.modules import datastore
from deepface.modules.database import milvus


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("similarity_search", [True, False])
def test_milvus_lite_distances_match_exact_search(
    tmp_path, monkeypatch, normalized, similarity_search
):
    sdk = pytest.importorskip("pymilvus")
    pytest.importorskip("milvus_lite")
    client = sdk.MilvusClient(uri=str(tmp_path / "distances.db"))
    adapter = milvus.MilvusClient(connection=client)
    monkeypatch.setattr(
        milvus, "build_model", lambda **kw: SimpleNamespace(output_shape=2)
    )
    criteria = dict(
        model_name="Facenet512",
        detector_backend="skip",
        aligned=True,
        l2_normalized=normalized,
    )
    adapter.initialize_database(**criteria)
    collection = client.list_collections()[0]
    vectors = (
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        if normalized
        else [[1.0, 0.0], [11.0, 0.0], [31.0, 0.0]]
    )
    try:
        client.insert(
            collection_name=collection,
            data=[
                dict(id=i, embedding=vector, img_name=f"image-{i}")
                for i, vector in enumerate(vectors)
            ],
        )
        raw = client.search(
            collection_name=collection,
            data=[[1.0, 0.0]],
            limit=3,
            search_params={"metric_type": "COSINE" if normalized else "L2"},
        )[0]
        raw_scores = {hit["id"]: hit["distance"] for hit in raw}
        assert [raw_scores[i] for i in range(3)] == pytest.approx(
            [1.0, 0.0, -1.0] if normalized else [0.0, 100.0, 900.0]
        )
        monkeypatch.setattr(datastore, "__connect_database", lambda **kw: adapter)
        monkeypatch.setattr(
            datastore,
            "represent",
            lambda **kw: [
                dict(embedding=[1.0, 0.0], facial_area=dict(x=0, y=0, w=1, h=1))
            ],
        )
        options = dict(
            img="synthetic",
            model_name="Facenet512",
            detector_backend="skip",
            database_type="milvus",
            connection=client,
            l2_normalize=normalized,
            distance_metric="cosine" if normalized else "euclidean",
            similarity_search=similarity_search,
            k=3,
        )
        exact = datastore.search(**options, search_method="exact")[0]
        ann = datastore.search(**options, search_method="ann")[0]
        assert ann["id"].tolist() == exact["id"].tolist()
        for column in ["distance", "threshold", "confidence"]:
            np.testing.assert_allclose(ann[column], exact[column], atol=1e-6)
        assert ann["distance_metric"].tolist() == exact["distance_metric"].tolist()
    finally:
        client.close()
