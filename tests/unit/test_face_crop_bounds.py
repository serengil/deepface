"""A detector box crossing the top or left border must crop the visible face."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from deepface.models.Detector import FacialAreaRegion
from deepface.models.face_detection.MtCnn import MtCnnClient
from deepface.models.face_detection.OpenCv import OpenCvClient
from deepface.models.face_detection.Ssd import SsdClient
from deepface.modules import detection


@pytest.mark.parametrize(
    "x,y,w,h", [(-4, 3, 12, 10), (3, -4, 10, 12), (-4, -4, 12, 12), (3, 4, 8, 9)]
)
def test_mtcnn_boxes_are_cropped_to_visible_image(monkeypatch, x, y, w, h):
    img = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    client = MtCnnClient.__new__(MtCnnClient)
    client.model = SimpleNamespace(
        detect_faces=lambda _: [
            {
                "box": (x, y, w, h),
                "confidence": 0.95,
                "keypoints": {"left_eye": (2, 5), "right_eye": (6, 5)},
            }
        ]
    )
    monkeypatch.setattr(detection.modeling, "build_model", lambda **kwargs: client)

    region = client.detect_faces(img)[0]
    face = detection.extract_face(
        facial_area=region,
        img=img,
        align=False,
        expand_percentage=0,
        width_border=0,
        height_border=0,
        detector_backend="mtcnn",
    )
    expected = img[max(0, y) : y + h, max(0, x) : x + w]
    np.testing.assert_array_equal(face.img, expected)

    faces = detection.extract_faces(
        img_path=img,
        detector_backend="mtcnn",
        align=False,
        color_face="bgr",
        normalize_face=False,
    )

    assert len(faces) == 1
    np.testing.assert_array_equal(faces[0]["face"], expected)
    assert faces[0]["facial_area"] == {
        "x": max(0, x),
        "y": max(0, y),
        "w": expected.shape[1],
        "h": expected.shape[0],
        "left_eye": (6, 5),
        "right_eye": (2, 5),
    }


def test_ssd_negative_box_uses_visible_face_and_real_eye_fallback(monkeypatch):
    img = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    client = SsdClient.__new__(SsdClient)
    client.model = Mock()
    client.model.forward.return_value = np.array(
        [[[[0, 1, 0.95, -0.2, 0.15, 0.4, 0.65]]]], dtype=np.float32
    )
    eye_detector = OpenCvClient.__new__(OpenCvClient)
    eye_detector.model = {"eye_detector": Mock()}
    eye_detector.model["eye_detector"].detectMultiScale.return_value = []
    monkeypatch.setattr(
        detection.modeling,
        "build_model",
        lambda task, model_name: client if model_name == "ssd" else eye_detector,
    )

    faces = detection.extract_faces(
        img_path=img,
        detector_backend="ssd",
        align=False,
        color_face="bgr",
        normalize_face=False,
    )

    assert len(faces) == 1
    np.testing.assert_array_equal(faces[0]["face"], img[3:13, :8])
    assert faces[0]["facial_area"] == {
        "x": 0,
        "y": 3,
        "w": 8,
        "h": 10,
        "left_eye": None,
        "right_eye": None,
    }


def test_default_alignment_restores_border_coordinates(monkeypatch):
    img = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    client = MtCnnClient.__new__(MtCnnClient)

    def infer(padded_img):
        assert padded_img.shape == (40, 40, 3)
        return [
            {
                "box": (13, 14, 8, 9),
                "confidence": 0.95,
                "keypoints": {"left_eye": (12, 15), "right_eye": (16, 15)},
            }
        ]

    client.model = SimpleNamespace(detect_faces=infer)
    monkeypatch.setattr(detection.modeling, "build_model", lambda **kwargs: client)
    faces = detection.extract_faces(
        img_path=img,
        detector_backend="mtcnn",
        color_face="bgr",
        normalize_face=False,
    )

    np.testing.assert_array_equal(faces[0]["face"], img[4:13, 3:11])
    assert faces[0]["facial_area"] == {
        "x": 3,
        "y": 4,
        "w": 8,
        "h": 9,
        "left_eye": (6, 5),
        "right_eye": (2, 5),
    }


@pytest.mark.parametrize("left_eye,right_eye", [((6, 4), (2, 4)), ((6, 6), (2, 4))])
def test_alignment_of_negative_box_matches_visible_box(left_eye, right_eye):
    img = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    kwargs = dict(
        img=img,
        align=True,
        expand_percentage=0,
        width_border=0,
        height_border=0,
        detector_backend="mtcnn",
    )
    expected = detection.extract_face(
        facial_area=FacialAreaRegion(
            x=0,
            y=0,
            w=8,
            h=10,
            left_eye=left_eye,
            right_eye=right_eye,
        ),
        **kwargs,
    )
    actual = detection.extract_face(
        facial_area=FacialAreaRegion(
            x=-4,
            y=-3,
            w=12,
            h=13,
            left_eye=left_eye,
            right_eye=right_eye,
        ),
        **kwargs,
    )

    np.testing.assert_array_equal(actual.img, expected.img)
    assert actual.facial_area == expected.facial_area


def test_in_bounds_expanded_box_keeps_existing_crop():
    img = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    face = detection.extract_face(
        facial_area=FacialAreaRegion(x=4, y=5, w=8, h=8),
        img=img,
        align=False,
        expand_percentage=50,
        width_border=0,
        height_border=0,
        detector_backend="opencv",
    )

    np.testing.assert_array_equal(face.img, img[3:15, 2:14])
    assert (
        face.facial_area.x,
        face.facial_area.y,
        face.facial_area.w,
        face.facial_area.h,
    ) == (
        2,
        3,
        12,
        12,
    )


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize(
    "x,y,w,h", [(-10, 3, 5, 10), (3, -10, 10, 5), (25, 3, 5, 10), (3, 25, 10, 5)]
)
def test_fully_outside_boxes_have_empty_crops_and_nonnegative_sizes(align, x, y, w, h):
    img = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    face = detection.extract_face(
        facial_area=FacialAreaRegion(x=x, y=y, w=w, h=h),
        img=img,
        align=align,
        expand_percentage=0,
        width_border=0,
        height_border=0,
        detector_backend="opencv",
    )

    assert face.img.size == 0
    assert face.facial_area.w >= 0
    assert face.facial_area.h >= 0
    assert face.facial_area.w == 0 or face.facial_area.h == 0


@pytest.mark.parametrize(
    "box,expected_box",
    [
        ((-10, 13, 5, 10), (-10, 3, 0, 10)),
        ((13, -10, 10, 5), (3, -10, 10, 0)),
        ((45, 13, 5, 10), (35, 3, 0, 10)),
        ((13, 45, 10, 5), (3, 35, 10, 0)),
    ],
)
def test_default_alignment_restores_outside_boxes_after_padding(
    monkeypatch, box, expected_box
):
    img = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    client = MtCnnClient.__new__(MtCnnClient)

    def infer(padded_img):
        assert padded_img.shape == (40, 40, 3)
        np.testing.assert_array_equal(padded_img[10:30, 10:30], img[:, :, ::-1])
        assert not padded_img[:10].any()
        return [
            {
                "box": box,
                "confidence": 0.95,
                "keypoints": {"left_eye": (12, 15), "right_eye": (16, 15)},
            }
        ]

    client.model = SimpleNamespace(detect_faces=infer)
    monkeypatch.setattr(detection.modeling, "build_model", lambda **kwargs: client)

    faces = detection.detect_faces(detector_backend="mtcnn", img=img)

    assert len(faces) == 1
    face = faces[0]
    assert face.img.size == 0
    area = face.facial_area
    assert (area.x, area.y, area.w, area.h) == expected_box
    assert area.left_eye == (6, 5)
    assert area.right_eye == (2, 5)
    assert (
        detection.extract_faces(
            img_path=img, detector_backend="mtcnn", enforce_detection=False
        )
        == []
    )
