"""A gallery entry without a face must not break batched recognition."""

import pickle

import cv2
import numpy as np
import pytest

from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import modeling, recognition
from deepface.modules.exceptions import SpoofDetected


class ColorModel(FacialRecognition):
    input_shape = (8, 8)
    output_shape = 3

    def forward(self, img):
        embeddings = img.mean(axis=(1, 2)).tolist()
        return embeddings[0] if len(embeddings) == 1 else embeddings


@pytest.mark.parametrize("missing", ["first", "last", "all", "none"])
@pytest.mark.parametrize("similarity_search", [False, True])
def test_batched_find_ignores_missing_embeddings(
    tmp_path, monkeypatch, missing, similarity_search
):
    image = np.full((8, 8, 3), [51, 102, 153], dtype=np.uint8)
    gallery_path = str(tmp_path / "gallery.png")
    assert cv2.imwrite(gallery_path, image)
    valid = {
        "identity": gallery_path,
        "hash": "cached",
        "embedding": [0.2, 0.4, 0.6],
        "target_x": 0,
        "target_y": 0,
        "target_w": 8,
        "target_h": 8,
    }
    invalid = dict(valid, identity=str(tmp_path / "no-face.png"), embedding=None)
    galleries = {
        "first": [invalid, valid],
        "last": [valid, invalid],
        "all": [invalid],
        "none": [valid],
    }
    cache = (
        tmp_path
        / "ds_model_facenet512_detector_skip_unaligned_normalization_base_expand_0.pkl"
    )
    cache.write_bytes(pickle.dumps(galleries[missing]))
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: ColorModel())

    results = recognition.find(
        image,
        str(tmp_path),
        model_name="Facenet512",
        detector_backend="skip",
        align=False,
        distance_metric="euclidean",
        threshold=0.01,
        refresh_database=False,
        batched=True,
        similarity_search=similarity_search,
        silent=True,
    )

    assert len(results) == 1
    if missing == "all":
        assert results == [[]]
    else:
        assert [item["identity"] for item in results[0]] == [gallery_path]
        assert results[0][0]["distance"] == pytest.approx(0, abs=1e-6)


def test_empty_valid_gallery_preserves_query_antispoofing():
    source = {"face": np.zeros((8, 8, 3)), "facial_area": {}, "is_real": False}
    with pytest.raises(SpoofDetected):
        recognition.find_batched(
            [{"embedding": None}], [source], anti_spoofing=True, threshold=0.5
        )


def test_filtered_gallery_keeps_query_order_and_top_k_metadata(monkeypatch):
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: ColorModel())
    gallery = [
        {"identity": "no-face", "embedding": None, "target_x": 0},
        {"identity": "second", "embedding": [0.8, 0.6, 0.4], "target_x": 20},
        {"identity": "first", "embedding": [0.2, 0.4, 0.6], "target_x": 10},
    ]
    sources = [
        {
            "face": np.full((8, 8, 3), color, dtype=np.uint8),
            "facial_area": {"x": x, "y": 0, "w": 8, "h": 8},
        }
        for x, color in [(1, [51, 102, 153]), (2, [204, 153, 102])]
    ]
    result = recognition.find_batched(
        gallery,
        sources,
        model_name="Facenet512",
        distance_metric="euclidean",
        threshold=0.01,
        k=1,
    )
    assert [rows[0]["identity"] for rows in result] == ["first", "second"]
    assert [rows[0]["target_x"] for rows in result] == [10, 20]
    assert [rows[0]["source_x"] for rows in result] == [1, 2]
    assert all(len(rows) == 1 for rows in result)
