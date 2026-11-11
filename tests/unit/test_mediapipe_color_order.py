"""The MediaPipe boundary consumes RGB while DeepFace detectors receive BGR."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from deepface.models.face_detection.external.MediaPipe import MediaPipeClient


def client_with(detections):
    client = MediaPipeClient.__new__(MediaPipeClient)
    client.model = Mock()
    client.model.process.return_value = SimpleNamespace(detections=detections)
    return client


def test_detector_receives_rgb_without_mutating_input():
    client = client_with(None)
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    original = bgr.copy()
    bgr.flags.writeable = False
    assert client.detect_faces(bgr) == []
    received = client.model.process.call_args.args[0]
    np.testing.assert_array_equal(received, [[[30, 20, 10], [60, 50, 40]]])
    np.testing.assert_array_equal(bgr, original)
    assert received.flags.c_contiguous


def test_color_conversion_keeps_landmarks_in_original_coordinates():
    detection = SimpleNamespace(
        score=[0.95],
        location_data=SimpleNamespace(
            relative_bounding_box=SimpleNamespace(
                xmin=0.1, ymin=0.2, width=0.4, height=0.5
            ),
            relative_keypoints=[
                SimpleNamespace(x=0.2, y=0.3),
                SimpleNamespace(x=0.4, y=0.3),
            ],
        ),
    )
    client = client_with([detection])
    faces = client.detect_faces(np.zeros((20, 40, 3), dtype=np.uint8)[:, ::-1])
    assert len(faces) == 1
    face = faces[0]
    assert (face.x, face.y, face.w, face.h) == (4, 4, 16, 10)
    assert face.right_eye == (8, 6)
    assert face.left_eye == (16, 6)
    assert face.confidence == 0.95
    assert client.model.process.call_args.args[0].flags.c_contiguous
