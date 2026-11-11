"""YuNet's score prefilter and OpenCV NMS use different threshold boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from deepface.models.face_detection.onnx.YuNet import INPUT_SIZE, STRIDES, YuNetClient


def detector(score):
    outputs = {}
    for stride in STRIDES:
        count = (INPUT_SIZE // stride) ** 2
        outputs[f"cls_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"obj_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"bbox_{stride}"] = np.zeros((1, count, 4), np.float32)
        outputs[f"kps_{stride}"] = np.zeros((1, count, 10), np.float32)
    outputs["cls_8"][0, 0, 0] = score
    outputs["obj_8"][0, 0, 0] = score
    client = YuNetClient.__new__(YuNetClient)
    client.model = Mock()
    client.model.get_inputs.return_value = [SimpleNamespace(name="input")]
    client.model.get_outputs.return_value = [
        SimpleNamespace(name=name) for name in outputs
    ]
    client.model.run.return_value = list(outputs.values())
    return client


def test_no_rows_survive_strict_nms_threshold():
    # The decode prefilter includes equality; NMSBoxes requires strictly greater scores.
    faces = detector(1.0).forward(np.zeros((32, 32, 3), np.uint8), 1.0)
    assert faces.shape == (0, 15)
    assert faces.dtype == np.float32


def test_no_rows_pass_the_decode_prefilter():
    assert detector(0.0).forward(np.zeros((32, 32, 3), np.uint8), 0.9).shape == (0, 15)


def test_surviving_detection_retains_its_landmarks_and_score():
    faces = detector(1.0).forward(np.zeros((32, 32, 3), np.uint8), 0.9)
    assert faces.shape == (1, 15)
    np.testing.assert_allclose(faces[0], [-4, -4, 8, 8] + [0] * 10 + [1])
