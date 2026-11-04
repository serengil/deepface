"""mtcnn runs on every backend engine, and its onnx port finds the same faces as pytorch."""

import cv2
import numpy as np
import pytest

from deepface.commons import backend_utils
from deepface.commons.logger import Logger
from deepface.models.face_detection.onnx import MtCnn as onnx_mtcnn
from deepface.modules import modeling


@pytest.mark.parametrize(
    "backend, module",
    [
        ("tensorflow", "deepface.models.face_detection.tf.MtCnn"),
        ("pytorch", "deepface.models.face_detection.pytorch.MtCnn"),
        ("onnx", "deepface.models.face_detection.onnx.MtCnn"),
    ],
)
def test_mtcnn_is_routed_to_backend_engine(monkeypatch, backend, module):
    monkeypatch.setattr(backend_utils, "get_backend_engine", lambda: backend)
    model_class = modeling.get_model_class(task="face_detector", model_name="mtcnn")
    assert model_class.__module__ == module
    assert model_class.__name__ == "MtCnnClient"


@pytest.mark.parametrize("backend", ["tensorflow", "pytorch", "onnx"])
def test_fastmtcnn_is_deprecated_alias_of_pytorch_mtcnn(monkeypatch, backend):
    messages = []
    monkeypatch.setattr(
        Logger, "warn", lambda self, message, show_once=False: messages.append(message)
    )
    monkeypatch.setattr(backend_utils, "get_backend_engine", lambda: backend)

    model_class = modeling.get_model_class(task="face_detector", model_name="fastmtcnn")

    assert model_class.__module__ == "deepface.models.face_detection.pytorch.MtCnn"
    assert len(messages) == 1
    assert "deprecated" in messages[0] and "pytorch" in messages[0]


@pytest.mark.parametrize(
    "in_shape, out_shape",
    [
        ((1588, 2280), (953, 1368)),  # first pyramid level of a large image
        ((1588, 2280), (16, 23)),  # last pyramid level - long bins
        ((500, 333), (24, 24)),  # downsampled candidate
        ((7, 5), (24, 24)),  # upsampled candidate
        ((1, 1), (48, 48)),
        ((24, 24), (24, 24)),
        ((101, 99), (17, 13)),
    ],
)
def test_area_resize_matches_torch(in_shape, out_shape):
    torch = pytest.importorskip("torch")
    img = np.random.default_rng(0).uniform(0, 255, size=(*in_shape, 3)).astype(np.float32)
    # crops are slices of the image, so non contiguous inputs must work too
    img = img[:, ::-1]

    expected = (
        torch.nn.functional.interpolate(
            torch.from_numpy(img.copy()).permute(2, 0, 1)[None],
            size=out_shape,
            mode="area",
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )
    actual = onnx_mtcnn.area_resize(img, *out_shape)

    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-3)


def test_nms_keeps_highest_scores_and_suppresses_overlaps():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float32
    )
    scores = np.array([0.8, 0.9, 0.7, 0.6], dtype=np.float32)

    assert onnx_mtcnn.nms(boxes, scores, 0.5).tolist() == [1, 2]
    assert onnx_mtcnn.nms(boxes, scores, 1.0).tolist() == [1, 0, 2, 3]


def test_nms_min_suppresses_box_inside_another():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.95], dtype=np.float32)

    # their iou is tiny, but the small box is entirely inside the large one
    assert onnx_mtcnn.nms(boxes, scores, 0.7).tolist() == [1, 0]
    assert onnx_mtcnn.nms_min(boxes, scores, 0.7).tolist() == [1]


class _NoFacePNet:  # pylint: disable=too-few-public-methods
    calls = 0

    def run(self, _outputs, inputs):
        self.calls += 1
        n, h, w, _ = inputs["input"].shape
        out_h, out_w = (h - 2) // 2 - 4, (w - 2) // 2 - 4
        out_h, out_w = max(out_h, 1), max(out_w, 1)
        probs = np.zeros((n, 2, out_h, out_w), dtype=np.float32)
        return [np.zeros((n, 4, out_h, out_w), dtype=np.float32), probs]


def test_no_candidates_and_tiny_images_return_no_faces():
    client = onnx_mtcnn.MtCnnClient.__new__(onnx_mtcnn.MtCnnClient)
    client.pnet = _NoFacePNet()

    assert not client.detect_faces(np.zeros((120, 160, 3), dtype=np.uint8))
    assert client.pnet.calls > 0

    # an image smaller than the minimum face size has no pyramid level at all
    client.pnet = _NoFacePNet()
    assert not client.detect_faces(np.zeros((15, 15, 3), dtype=np.uint8))
    assert client.pnet.calls == 0


@pytest.mark.parametrize(
    "img_path",
    [
        "dataset/img1.jpg",
        "dataset/img11.jpg",
        "dataset/couple.jpg",
        "dataset/selfie-many-people.jpg",
    ],
)
def test_onnx_mtcnn_finds_same_faces_as_pytorch(img_path):
    pytest.importorskip("facenet_pytorch")
    from deepface.models.face_detection.pytorch.MtCnn import MtCnnClient as TorchMtCnn

    img = cv2.imread(img_path)
    assert img is not None, img_path

    expected = TorchMtCnn().detect_faces(img)
    actual = onnx_mtcnn.MtCnnClient().detect_faces(img)

    assert len(expected) > 0
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert (a.x, a.y, a.w, a.h) == (e.x, e.y, e.w, e.h)
        assert a.left_eye == e.left_eye
        assert a.right_eye == e.right_eye
        assert a.confidence == pytest.approx(float(e.confidence), abs=1e-5)
