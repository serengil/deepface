"""Geometry and score order of CenterFace's dense heatmap decoder."""

import numpy as np

from deepface.models.face_detection.onnx.CenterFace import CenterFace


def test_edge_clipping_does_not_translate_box_or_eyes():
    decoder = CenterFace(session=None)
    scores = np.zeros((1, 1, 4, 4), np.float32)
    scores[0, 0, 0, 0] = 0.9
    scales = np.full((1, 2, 4, 4), np.log(4), np.float32)
    offsets = np.zeros_like(scales)
    points = np.full((1, 10, 4, 4), 0.5, np.float32)
    boxes, landmarks = decoder.decode(scores, scales, offsets, points, (16, 16))
    # The full prediction spans [-6, 10] in both axes and is centred at (2, 2).
    np.testing.assert_allclose(boxes, [[0, 0, 10, 10, 0.9]], atol=1e-5)
    np.testing.assert_allclose(landmarks, [[2] * 10], atol=1e-5)


def test_suppressed_proposal_does_not_decode_its_landmarks():
    decoder = CenterFace(session=None)
    scores = np.zeros((1, 1, 8, 8), np.float32)
    scores[0, 0, 3, 3:5] = [0.9, 0.8]
    scales = np.full((1, 2, 8, 8), np.log(4), np.float32)
    offsets = np.zeros_like(scales)
    landmarks = np.full((1, 10, 8, 8), 0.5, np.float32)
    landmarks[0, :, 3, 4] = np.finfo(np.float32).max
    with np.errstate(over="raise"):
        boxes, points = decoder.decode(scores, scales, offsets, landmarks, (32, 32))
    np.testing.assert_allclose(boxes, [[6, 6, 22, 22, 0.9]], atol=1e-5)
    np.testing.assert_allclose(points, [[14] * 10], atol=1e-5)


def test_nms_preserves_inclusive_pixel_overlap_and_score_order():
    decoder = CenterFace(session=None)
    boxes = np.array([[1, 1, 3, 3], [1, 1, 3, 3], [8, 8, 9, 9], [2, 1, 4, 3]], np.float32)
    scores = np.array([0.7, 0.9, 0.8, 0.6], np.float32)
    assert decoder.nms(boxes, scores, 0.5) == [1, 2]
    assert decoder.nms(boxes, scores, 0.51) == [1, 2, 3]
