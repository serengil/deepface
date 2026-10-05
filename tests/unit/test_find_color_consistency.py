"""Query and gallery crops must reach represent in the same color order."""

import cv2
import numpy as np
import pytest

from deepface.models.Detector import DetectedFace, FacialAreaRegion
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import detection, modeling, recognition


class RecordingModel(FacialRecognition):
    input_shape = (8, 8)
    output_shape = 3

    def __init__(self):
        self.inputs = []

    def forward(self, img):
        self.inputs.extend(img.copy())
        embeddings = img.mean(axis=(1, 2)).tolist()
        return embeddings[0] if len(embeddings) == 1 else embeddings


@pytest.mark.parametrize("return_type", ["pandas", "dict"])
@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("face_count", [1, 2])
def test_find_query_and_gallery_colors(
    tmp_path, monkeypatch, return_type, as_array, face_count
):
    image = np.full((8, 8 * face_count, 3), [20, 80, 220], dtype=np.uint8)
    image_path = str(tmp_path / "color.png")
    assert cv2.imwrite(image_path, image)
    model = RecordingModel()
    monkeypatch.setattr(modeling, "build_model", lambda **kw: model)
    backend = "skip"
    if face_count == 2:
        backend = "opencv"

        def detect_faces(**kwargs):
            img = kwargs["img"]
            return [
                DetectedFace(
                    img=img[:, x : x + 8],
                    facial_area=FacialAreaRegion(x=x, y=0, w=8, h=8),
                    confidence=1.0,
                )
                for x in [0, 8]
            ]

        monkeypatch.setattr(detection, "detect_faces", detect_faces)

    options = dict(
        img_path=image if as_array else image_path,
        db_path=str(tmp_path),
        model_name="Facenet512",
        detector_backend=backend,
        align=False,
        distance_metric="euclidean",
        threshold=0.01,
        return_type=return_type,
        silent=True,
    )
    results = recognition.find(**options)
    assert len(model.inputs) == 2 * face_count
    for gallery, query in zip(model.inputs[:face_count], model.inputs[face_count:]):
        np.testing.assert_array_equal(query, gallery)
        np.testing.assert_allclose(
            query[0, 0], np.array([20, 80, 220]) / 255, atol=1e-7
        )
    assert len(results) == face_count
    for result in results:
        rows = result if return_type == "dict" else result.to_dict("records")
        assert len(rows) == face_count
        assert all(
            row["identity"] == image_path and row["distance"] == 0 for row in rows
        )

    # A cached gallery must obey the same contract without rebuilding embeddings.
    model.inputs.clear()
    cached = recognition.find(**options, refresh_database=False)
    assert len(model.inputs) == face_count
    assert all(len(result) == face_count for result in cached)
