"""CenterFace must treat a heatmap below threshold as an ordinary no-face result."""

from unittest.mock import Mock

import numpy as np
import pytest

from deepface.models.face_detection.CenterFace import CenterFace, CenterFaceClient
from deepface.modules import detection
from deepface.modules.exceptions import FaceNotDetected


def make_model(score):
    model = CenterFace.__new__(CenterFace)
    heatmap = np.full((1, 1, 8, 8), score, dtype=np.float32)
    scale = np.zeros((1, 2, 8, 8), dtype=np.float32)
    offset = np.zeros_like(scale)
    landmarks = np.full((1, 10, 8, 8), 0.5, dtype=np.float32)
    model.net = Mock()
    model.net.forward.return_value = heatmap, scale, offset, landmarks
    return model


@pytest.mark.parametrize("score", [0.0, 0.35])
def test_empty_centerface_output(score):
    model = make_model(score)
    boxes, landmarks = model.forward(
        np.zeros((32, 32, 3), dtype=np.uint8), 32, 32, 0.35
    )

    assert boxes.shape == (0, 5)
    assert landmarks.shape == (0, 10)
    assert boxes.dtype == landmarks.dtype == np.float32


def test_centerface_positive_output_is_preserved():
    model = make_model(0.0)
    model.net.forward.return_value[0][0, 0, 0, 0] = 0.9
    boxes, landmarks = model.forward(
        np.zeros((32, 32, 3), dtype=np.uint8), 32, 32, 0.35
    )

    np.testing.assert_allclose(boxes, [[0, 0, 4, 4, 0.9]])
    np.testing.assert_allclose(landmarks, [[2] * 10])


@pytest.mark.parametrize("enforce_detection", [False, True])
def test_centerface_no_face_respects_enforce_detection(monkeypatch, enforce_detection):
    model = make_model(0.0)
    client = CenterFaceClient()
    monkeypatch.setattr(client, "build_model", lambda: model)
    monkeypatch.setattr(detection.modeling, "build_model", lambda **kwargs: client)
    img = np.full((32, 32, 3), 42, dtype=np.uint8)
    kwargs = dict(
        img_path=img,
        detector_backend="centerface",
        align=False,
        enforce_detection=enforce_detection,
        color_face="bgr",
        normalize_face=False,
    )

    if enforce_detection:
        with pytest.raises(FaceNotDetected):
            detection.extract_faces(**kwargs)
    else:
        faces = detection.extract_faces(**kwargs)
        assert len(faces) == 1
        assert faces[0]["confidence"] == 0
        np.testing.assert_array_equal(faces[0]["face"], img)
