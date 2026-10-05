"""An explicitly supplied zero threshold is distinct from the default."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from deepface.modules import modeling, recognition, verification
from deepface.models.FacialRecognition import FacialRecognition


@pytest.mark.parametrize("threshold", [None, 0, 0.0, 0.1, 0.2, 0.3])
@pytest.mark.parametrize("distance", [0.0, 0.2])
@pytest.mark.parametrize("metric", ["euclidean", verification.find_euclidean_distance])
def test_verify_threshold(monkeypatch, threshold, distance, metric):
    monkeypatch.setattr(
        modeling, "build_model", lambda **kw: SimpleNamespace(output_shape=2)
    )
    options = dict(
        img1_path=[1.0, 0.0],
        img2_path=[1.0, distance],
        model_name="Facenet512",
        distance_metric=metric,
        threshold=threshold,
    )
    if callable(metric) and threshold is None:
        with pytest.raises(ValueError, match="Threshold must be specified"):
            verification.verify(**options)
        return
    result = verification.verify(**options)
    expected = (
        verification.find_threshold("Facenet512", metric)
        if threshold is None
        else threshold
    )
    assert result["threshold"] == expected
    assert result["distance"] == distance
    assert result["verified"] == (distance <= expected)
    if result["verified"]:
        assert 51 <= result["confidence"] <= 100
    else:
        assert 0 <= result["confidence"] <= 49


class ColorModel(FacialRecognition):
    input_shape = (4, 4)
    output_shape = 2

    def forward(self, img):
        embeddings = [[1.0, round(float(face[:, :, 0].mean()), 6)] for face in img]
        return embeddings[0] if len(embeddings) == 1 else embeddings


@pytest.mark.parametrize("threshold", [None, 0, 0.0, 0.1, 0.2, 0.3])
@pytest.mark.parametrize("return_type", ["pandas", "dict"])
@pytest.mark.parametrize("metric", ["euclidean", verification.find_euclidean_distance])
def test_find_threshold(tmp_path, monkeypatch, threshold, return_type, metric):
    monkeypatch.setattr(modeling, "build_model", lambda **kw: ColorModel())
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    same = str(tmp_path / "same.png")
    different = str(tmp_path / "different.png")
    assert cv2.imwrite(same, image)
    other = image.copy()
    other[:, :, 0] = 51
    assert cv2.imwrite(different, other)
    options = dict(
        img_path=image,
        db_path=str(tmp_path),
        model_name="Facenet512",
        detector_backend="skip",
        align=False,
        distance_metric=metric,
        threshold=threshold,
        return_type=return_type,
        silent=True,
    )
    if callable(metric) and threshold is None:
        with pytest.raises(ValueError, match="Threshold must be specified"):
            recognition.find(**options)
        return
    result = recognition.find(**options)[0]
    rows = result if return_type == "dict" else result.to_dict("records")
    expected = (
        verification.find_threshold("Facenet512", metric)
        if threshold is None
        else threshold
    )
    assert [row["identity"] for row in rows] == (
        [same, different] if expected >= 0.2 else [same]
    )
    assert all(row["threshold"] == expected for row in rows)
    assert rows[0]["distance"] == 0.0
    if return_type == "pandas":
        assert all(51 <= row["confidence"] <= 100 for row in rows)


@pytest.mark.parametrize("threshold", [23.99, 24.0, 24.01])
def test_verify_confidence_with_looser_threshold(monkeypatch, threshold):
    monkeypatch.setattr(
        modeling, "build_model", lambda **kw: SimpleNamespace(output_shape=2)
    )
    result = verification.verify(
        [1.0, 0.0],
        [1.0, 24.0],
        model_name="Facenet512",
        distance_metric="euclidean",
        threshold=threshold,
        silent=True,
    )
    assert result["distance"] == 24.0
    assert result["verified"] == (24.0 <= threshold)
    if result["verified"]:
        assert 51 <= result["confidence"] <= 100
    else:
        assert 0 <= result["confidence"] <= 49
