"""Concurrent CenterFace requests must retain their own resize geometry."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from deepface.models.face_detection.onnx.CenterFace import CenterFace, CenterFaceClient
from deepface.modules import detection


def make_outputs(_output_names, inputs):
    """Supply one interior proposal in the network's actual output grid."""
    _, _, height, width = inputs["input"].shape
    height, width = height // 4, width // 4
    heatmap = np.zeros((1, 1, height, width), dtype=np.float32)
    heatmap[0, 0, height // 2, width // 2] = 0.9
    scale = np.full((1, 2, height, width), np.log(4), dtype=np.float32)
    offset = np.zeros_like(scale)
    landmarks = np.full((1, 10, height, width), 0.5, dtype=np.float32)
    return heatmap, scale, offset, landmarks


def make_model():
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="input")]
    session.run.side_effect = make_outputs
    return CenterFace(session=session)


def make_image(shape):
    height, width = shape
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def forward(model, image):
    return model.forward(image, image.shape[0], image.shape[1], threshold=0.35)


def run_overlapping(model, images, process):
    barrier = Barrier(2)

    def infer(output_names, inputs):
        outputs = make_outputs(output_names, inputs)
        # Both requests finish preparing their own input before either decodes it.
        barrier.wait(timeout=5)
        return outputs

    with patch.object(model.session, "run", side_effect=infer):
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(process, image) for image in images]
            return [future.result(timeout=5) for future in pending]


@pytest.mark.parametrize(
    "shapes",
    [((33, 49), (63, 61)), ((33, 65), (81, 37))],
)
def test_overlapping_requests_match_sequential_geometry(shapes):
    model = make_model()
    images = [make_image(shape) for shape in shapes]
    expected = [forward(model, image) for image in images]

    actual = run_overlapping(model, images, lambda image: forward(model, image))

    for (boxes, landmarks), (expected_boxes, expected_landmarks) in zip(actual, expected):
        assert boxes.shape == (1, 5)
        assert landmarks.shape == (1, 10)
        np.testing.assert_array_equal(boxes, expected_boxes)
        np.testing.assert_array_equal(landmarks, expected_landmarks)


def test_overlapping_extraction_keeps_the_correct_face_pixels():
    model = make_model()
    with patch.object(CenterFaceClient, "build_model", return_value=model):
        client = CenterFaceClient()
    images = [make_image((33, 65)), make_image((81, 37))]

    def extract(image):
        return detection.extract_faces(
            img_path=image,
            detector_backend="centerface",
            align=False,
            color_face="bgr",
            normalize_face=False,
        )

    with patch.object(detection.modeling, "build_model", return_value=client):
        expected = [extract(image) for image in images]
        actual = run_overlapping(model, images, extract)

    for faces, expected_faces in zip(actual, expected):
        assert len(faces) == len(expected_faces) == 1
        assert faces[0]["facial_area"] == expected_faces[0]["facial_area"]
        assert faces[0]["confidence"] == expected_faces[0]["confidence"]
        np.testing.assert_array_equal(faces[0]["face"], expected_faces[0]["face"])


def test_sequential_requests_can_alternate_image_sizes():
    model = make_model()
    images = [make_image((33, 65)), make_image((81, 37))]
    expected = [forward(model, image) for image in images]

    for image, reference in zip(images * 2, expected * 2):
        actual = forward(model, image)
        for result, expected_result in zip(actual, reference):
            np.testing.assert_array_equal(result, expected_result)
