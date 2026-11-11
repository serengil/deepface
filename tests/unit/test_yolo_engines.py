"""yolo runs on pytorch and onnx, and its onnx port finds the same faces as pytorch."""

import cv2
import numpy as np
import pytest

from deepface.commons import backend_utils
from deepface.models.face_detection.onnx import Yolo as onnx_yolo
from deepface.modules import modeling

YOLO_MODELS = {
    "yolov8n": "V8n",
    "yolov8m": "V8m",
    "yolov8l": "V8l",
    "yolov11n": "V11n",
    "yolov11s": "V11s",
    "yolov11m": "V11m",
    "yolov11l": "V11l",
    "yolov12n": "V12n",
    "yolov12s": "V12s",
    "yolov12m": "V12m",
    "yolov12l": "V12l",
}


@pytest.mark.parametrize("model_name, suffix", YOLO_MODELS.items())
@pytest.mark.parametrize(
    "backend, module",
    [
        ("pytorch", "deepface.models.face_detection.pytorch.Yolo"),
        ("onnx", "deepface.models.face_detection.onnx.Yolo"),
        # tensorflow has no yolo of its own, it falls back to the onnx one
        ("tensorflow", "deepface.models.face_detection.onnx.Yolo"),
    ],
)
def test_yolo_is_routed_to_backend_engine(monkeypatch, backend, module, model_name, suffix):
    monkeypatch.setattr(backend_utils, "get_backend_engine", lambda: backend)
    model_class = modeling.get_model_class(task="face_detector", model_name=model_name)
    assert model_class.__module__ == module
    assert model_class.__name__ == f"YoloDetectorClient{suffix}"


def test_onnx_and_pytorch_cover_same_models():
    from deepface.models.face_detection.pytorch.Yolo import YoloModel as TorchYoloModel

    assert [m.name for m in onnx_yolo.YoloModel] == [m.name for m in TorchYoloModel]
    for model in onnx_yolo.YoloModel:
        torch_file = TorchYoloModel[model.name].file_name
        assert model.file_name == torch_file.replace(".pt", ".onnx")
        assert all(url.endswith(model.file_name) for url in model.url)


def test_input_sizes_are_the_ones_checkpoints_are_trained_with():
    sizes = {model.name: model.input_size for model in onnx_yolo.YoloModel}
    assert sizes.pop("V8M") == 960
    assert sizes.pop("V8L") == 960
    assert set(sizes.values()) == {640}


@pytest.mark.parametrize("size", [640, 960])
@pytest.mark.parametrize(
    "shape", [(640, 640), (480, 640), (1080, 1920), (1920, 1080), (333, 500), (100, 50)]
)
def test_letterbox_pads_to_stride(shape, size):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    padded, gain = onnx_yolo.letterbox(img, size)
    assert max(padded.shape[:2]) == size
    assert padded.shape[0] % onnx_yolo.STRIDE == 0
    assert padded.shape[1] % onnx_yolo.STRIDE == 0
    assert gain == pytest.approx(size / max(shape))


def test_nms_keeps_highest_scores_and_suppresses_overlaps():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float32
    )
    scores = np.array([0.8, 0.9, 0.7, 0.6], dtype=np.float32)

    assert onnx_yolo.nms(boxes, scores, 0.5).tolist() == [1, 2]
    assert onnx_yolo.nms(boxes, scores, 1.0).tolist() == [1, 0, 2, 3]


class _NoFaceSession:  # pylint: disable=too-few-public-methods
    class _Input:  # pylint: disable=too-few-public-methods
        name = "images"

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, inputs):
        _, _, h, w = inputs["images"].shape
        anchors = sum((h // s) * (w // s) for s in (8, 16, 32))
        return [np.zeros((1, 5, anchors), dtype=np.float32)]


def test_no_candidates_return_no_faces():
    client = onnx_yolo.YoloDetectorClient.__new__(onnx_yolo.YoloDetectorClient)
    client.model = _NoFaceSession()
    client.input_size = 640
    assert not client.detect_faces(np.zeros((120, 160, 3), dtype=np.uint8))


@pytest.mark.parametrize("model_name", YOLO_MODELS.keys())
def test_onnx_yolo_finds_same_faces_as_pytorch(monkeypatch, model_name):
    pytest.importorskip("ultralytics")

    monkeypatch.setattr(backend_utils, "get_backend_engine", lambda: "pytorch")
    torch_detector = modeling.get_model_class(task="face_detector", model_name=model_name)()
    monkeypatch.setattr(backend_utils, "get_backend_engine", lambda: "onnx")
    onnx_detector = modeling.get_model_class(task="face_detector", model_name=model_name)()

    for img_path in [
        "dataset/img1.jpg",
        "dataset/img11.jpg",
        "dataset/couple.jpg",
        "dataset/selfie-many-people.jpg",
    ]:
        img = cv2.imread(img_path)
        assert img is not None, img_path

        expected = torch_detector.detect_faces(img)
        actual = onnx_detector.detect_faces(img)

        assert len(expected) > 0, img_path
        assert len(actual) == len(expected), img_path
        for a, e in zip(actual, expected):
            # onnxruntime and torch accumulate floats in different orders, so a coordinate
            # may be truncated into the neighbouring integer
            assert np.abs(np.array([a.x, a.y, a.w, a.h]) - [e.x, e.y, e.w, e.h]).max() <= 1
            assert a.confidence == pytest.approx(float(e.confidence), abs=1e-4)
            if e.left_eye is None:
                assert a.left_eye is None and a.right_eye is None
            else:
                assert np.abs(np.array(a.left_eye) - e.left_eye).max() <= 1
                assert np.abs(np.array(a.right_eye) - e.right_eye).max() <= 1
