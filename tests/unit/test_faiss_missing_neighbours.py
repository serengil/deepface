"""FAISS padding is not an identity when the requested gallery is too small."""

from unittest.mock import MagicMock

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from deepface.modules import datastore


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("similarity_search", [False, True])
@pytest.mark.parametrize("k", [None, 1, 4])
def test_ann_search_omits_missing_neighbours(
    monkeypatch, normalized, similarity_search, k
):
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0] if normalized else [101.0, 0.0]],
        dtype="float32",
    )
    index = faiss.IndexIDMap(faiss.IndexHNSWFlat(2, 32))
    index.add_with_ids(vectors, np.array([42, 7], dtype="int64"))
    storage = MagicMock()
    storage.get_embeddings_index.return_value = faiss.serialize_index(index).tobytes()
    storage.search_by_id.side_effect = lambda ids: [
        {"id": identity, "img_name": f"image-{identity}"} for identity in ids
    ]
    monkeypatch.setattr(datastore, "__connect_database", lambda **kw: storage)
    monkeypatch.setattr(
        datastore, "represent", lambda **kw: [{"embedding": [1.0, 0.0]}]
    )

    frames = datastore.search(
        img="synthetic",
        model_name="Facenet512",
        detector_backend="skip",
        l2_normalize=normalized,
        similarity_search=similarity_search,
        k=k,
        search_method="ann",
        database_type="postgres",
        connection=storage,
    )

    expected = [42, 7] if similarity_search and k != 1 else [42]
    assert frames[0]["id"].tolist() == expected
    assert frames[0]["img_name"].tolist() == [f"image-{i}" for i in expected]
    storage.search_by_id.assert_called_once_with(ids=expected)
