"""Similarity retrieval must not turn every ranked candidate into a verified match."""

import numpy as np
import pytest

from deepface.modules import modeling, recognition, verification

CACHE_NAME = "ds_model_sface_detector_skip_aligned_normalization_base_expand_0.npz"


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    class FixedEmbedding:
        input_shape = (8, 8)
        output_shape = 2

        def forward(self, _images):
            return [1.0, 0.0]

    monkeypatch.setattr(modeling, "build_model", lambda **_kwargs: FixedEmbedding())
    rows = [
        dict(
            identity=name,
            hash="cached-" + name,
            embedding=embedding,
            target_x=0,
            target_y=0,
            target_w=8,
            target_h=8,
        )
        for name, embedding in [
            ("same.png", [1.0, 0.0]),
            ("near.png", [0.8, 0.6]),
            ("far.png", [-20.0, 0.0]),
        ]
    ]
    (tmp_path / CACHE_NAME).write_bytes(recognition.__serialize_npz(rows))
    return tmp_path


@pytest.mark.parametrize("similarity_search", [False, True], ids=["verify", "similarity"])
@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean"])
@pytest.mark.parametrize("threshold", [None, 0.0, 0.1], ids=["default", "zero", "strict"])
@pytest.mark.parametrize("k", [None, 2], ids=["all", "top-two"])
def test_similarity_confidence_matches_dataframe_results(
    gallery, similarity_search, distance_metric, threshold, k
):
    options = dict(
        img_path=np.full((8, 8, 3), 128, dtype=np.uint8),
        db_path=str(gallery),
        model_name="SFace",
        detector_backend="skip",
        enforce_detection=False,
        align=True,
        refresh_database=False,
        silent=True,
        similarity_search=similarity_search,
        distance_metric=distance_metric,
        threshold=threshold,
        k=k,
    )
    dictionaries = recognition.find(**options, return_type="dict")[0]
    dataframe = recognition.find(**options, return_type="pandas")[0].to_dict("records")
    effective_threshold = (
        verification.find_threshold("SFace", distance_metric) if threshold is None else threshold
    )

    assert [row["identity"] for row in dictionaries] == [row["identity"] for row in dataframe]
    if similarity_search:
        assert len(dictionaries) == (3 if k is None else k)
        # Preserve the existing unbounded-retrieval threshold in dictionary metadata.
        assert all(np.isinf(row["threshold"]) for row in dictionaries)
    else:
        assert all(row["threshold"] == effective_threshold for row in dictionaries)
    assert all(row["threshold"] == effective_threshold for row in dataframe)

    for dictionary_row, dataframe_row in zip(dictionaries, dataframe):
        assert dictionary_row["distance"] == pytest.approx(dataframe_row["distance"])
        assert dictionary_row["confidence"] == pytest.approx(dataframe_row["confidence"])
        verified = dictionary_row["distance"] <= effective_threshold
        if verified:
            assert 51 <= dictionary_row["confidence"] <= 100
        else:
            assert 0 <= dictionary_row["confidence"] <= 49
