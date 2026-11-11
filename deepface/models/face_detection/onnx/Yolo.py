"""
Yolo face detectors running on onnxruntime, without ultralytics and torch.

The graphs are the pytorch checkpoints of deepface.models.face_detection.pytorch.Yolo exported
to onnx with ultralytics (opset 17, dynamic input shape). The pre and post processing here is a
numpy port of the ultralytics predict pipeline - rect letterbox, nms and rescaling - so the onnx
backend engine finds the same faces as the pytorch one.

References - the pytorch checkpoints these graphs are exported from
    - yolov8n-face: https://github.com/derronqi/yolov8-face
    - yolov8m-face, yolov8l-face, yolov11 and yolov12 face models:
        https://github.com/akanametov/yolo-face (formerly github.com/YapaLab/yolo-face)
    - model architectures and training framework: https://github.com/ultralytics/ultralytics

!!! LICENSE !!!
    - derronqi/yolov8-face is licensed under GPL-3.0
        https://github.com/derronqi/yolov8-face/blob/main/LICENSE
    - akanametov/yolo-face is licensed under GPL-3.0
        https://github.com/akanametov/yolo-face/blob/main/LICENSE
    - ultralytics is licensed under AGPL-3.0
        https://github.com/ultralytics/ultralytics/blob/main/LICENSE
    Exporting the weights to onnx does not change their license. Unlike the rest of deepface,
    which is MIT licensed, these models are copyleft. Check their license terms before you use
    them in a commercial or closed source product.
"""

# built-in dependencies
import os
from enum import Enum
from typing import Any, List, Optional, Tuple

# 3rd party dependencies
import cv2
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

# defaults of ultralytics' predict, which the pytorch backend uses. the input size is the
# image size a checkpoint is trained with, ultralytics predicts with it - see YoloModel
STRIDE = 32
PADDING_VALUE = 114
DEFAULT_CONFIDENCE = 0.25
IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300
MAX_NMS = 30000

# yolov8n-face is a pose model, it predicts 5 keypoints - right eye, left eye, nose and
# mouth corners - each as (x, y, visibility). other models, yolov8m and yolov8l included,
# predict boxes only.
NUM_KEYPOINTS = 5


def _sources(file_name: str) -> List[str]:
    return [
        f"https://github.com/serengil/deepface_models/releases/download/v1.0/{file_name}",
        f"https://huggingface.co/serengil/deepface/resolve/main/{file_name}",
    ]


class YoloModel(Enum):
    # file name, source urls, input size - yolov8m and yolov8l are trained with 960x960 images
    V8N = ("yolov8n-face.onnx", _sources("yolov8n-face.onnx"), 640)
    V8M = ("yolov8m-face.onnx", _sources("yolov8m-face.onnx"), 960)
    V8L = ("yolov8l-face.onnx", _sources("yolov8l-face.onnx"), 960)
    V11N = ("yolov11n-face.onnx", _sources("yolov11n-face.onnx"), 640)
    V11S = ("yolov11s-face.onnx", _sources("yolov11s-face.onnx"), 640)
    V11M = ("yolov11m-face.onnx", _sources("yolov11m-face.onnx"), 640)
    V11L = ("yolov11l-face.onnx", _sources("yolov11l-face.onnx"), 640)
    V12N = ("yolov12n-face.onnx", _sources("yolov12n-face.onnx"), 640)
    V12S = ("yolov12s-face.onnx", _sources("yolov12s-face.onnx"), 640)
    V12M = ("yolov12m-face.onnx", _sources("yolov12m-face.onnx"), 640)
    V12L = ("yolov12l-face.onnx", _sources("yolov12l-face.onnx"), 640)

    def __init__(self, file_name: str, url: List[str], input_size: int):
        self.file_name = file_name
        self.url = url
        self.input_size = input_size


def letterbox(img: NDArray[Any], size: int) -> Tuple[NDArray[Any], float]:
    """
    Resize an image to fit into a size x size square keeping its aspect ratio, and pad it to
    the smallest multiple of STRIDE - ultralytics' LetterBox with auto=True
    Args:
        img (np.ndarray): image in bgr
        size (int): input size of the model
    Returns:
        padded (np.ndarray): resized and padded image
        gain (float): resize factor
    """
    height, width = img.shape[:2]
    gain = min(size / height, size / width)
    new_width, new_height = int(round(width * gain)), int(round(height * gain))
    dw = ((size - new_width) % STRIDE) / 2
    dh = ((size - new_height) % STRIDE) / 2

    if (width, height) != (new_width, new_height):
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PADDING_VALUE,) * 3
    )
    return padded, gain


def nms(boxes: NDArray[Any], scores: NDArray[Any], threshold: float) -> NDArray[Any]:
    """
    Greedy non maximum suppression - torchvision.ops.nms
    Args:
        boxes (np.ndarray): (n, 4) shaped boxes in x1, y1, x2, y2
        scores (np.ndarray): (n,) shaped scores
        threshold (float): boxes overlapping a kept one more than this iou are dropped
    Returns:
        keep (np.ndarray): indices of the kept boxes, sorted by score in descending order
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= threshold]
    return np.array(keep, dtype=np.int64)


class YoloDetectorClient(Detector):
    def __init__(self, model: YoloModel):
        super().__init__()
        self.input_size = model.input_size
        self.model = self.build_model(model)

    def build_model(self, model: YoloModel) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (onnxruntime.InferenceSession)
        """
        return build_session(file_name=model.file_name, source_url=model.url)

    def detect_faces(self, img: NDArray[Any]) -> List[FacialAreaRegion]:
        """
        Detect and align face with yolo

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        confidence_threshold = float(
            os.getenv("YOLO_MIN_DETECTION_CONFIDENCE", str(DEFAULT_CONFIDENCE))
        )
        img_height, img_width = img.shape[:2]

        padded, gain = letterbox(img, self.input_size)
        blob = padded[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        input_name = self.model.get_inputs()[0].name
        # (1, 4 + 1 + keypoints * 3, anchors) -> (anchors, 4 + 1 + keypoints * 3)
        predictions = self.model.run(None, {input_name: np.ascontiguousarray(blob)})[0][0].T

        predictions = predictions[predictions[:, 4] > confidence_threshold]
        if predictions.shape[0] == 0:
            return []
        if predictions.shape[0] > MAX_NMS:
            predictions = predictions[np.argsort(-predictions[:, 4], kind="stable")[:MAX_NMS]]

        # center x, center y, width, height to x1, y1, x2, y2
        boxes = np.empty((predictions.shape[0], 4), dtype=np.float32)
        boxes[:, :2] = predictions[:, :2] - predictions[:, 2:4] / 2
        boxes[:, 2:] = predictions[:, :2] + predictions[:, 2:4] / 2
        scores = predictions[:, 4]

        keep = nms(boxes, scores, IOU_THRESHOLD)[:MAX_DETECTIONS]
        boxes, scores, predictions = boxes[keep], scores[keep], predictions[keep]

        # map boxes back to the original image - ultralytics' scale_boxes
        padded_height, padded_width = padded.shape[:2]
        pad_x = round((padded_width - img_width * gain) / 2 - 0.1)
        pad_y = round((padded_height - img_height * gain) / 2 - 0.1)
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / gain).clip(0, img_width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / gain).clip(0, img_height)

        keypoints: Optional[NDArray[Any]] = None
        if predictions.shape[1] == 5 + NUM_KEYPOINTS * 3:
            # ultralytics' scale_coords does not round the padding
            kpts = predictions[:, 5:].reshape(-1, NUM_KEYPOINTS, 3)[:, :, :2].copy()
            kpts[:, :, 0] = ((kpts[:, :, 0] - (padded_width - img_width * gain) / 2) / gain).clip(
                0, img_width
            )
            kpts[:, :, 1] = ((kpts[:, :, 1] - (padded_height - img_height * gain) / 2) / gain).clip(
                0, img_height
            )
            keypoints = kpts

        # x1, y1, x2, y2 to center x, center y, width, height in float32 as ultralytics does
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        sizes = boxes[:, 2:] - boxes[:, :2]

        resp = []
        for idx in range(boxes.shape[0]):
            x, y = centers[idx].tolist()
            w, h = sizes[idx].tolist()

            right_eye: Optional[Tuple[int, int]] = None
            left_eye: Optional[Tuple[int, int]] = None

            # yolov8n-face is detecting eyes through keypoints, while others do not
            if keypoints is not None:
                (rx, ry), (lx, ly) = keypoints[idx, :2].tolist()
                right_eye = (int(rx), int(ry))
                left_eye = (int(lx), int(ly))

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=float(scores[idx]),
            )
            resp.append(facial_area)

        return resp


class YoloDetectorClientV8n(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V8N)


class YoloDetectorClientV8m(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V8M)


class YoloDetectorClientV8l(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V8L)


class YoloDetectorClientV11n(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V11N)


class YoloDetectorClientV11s(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V11S)


class YoloDetectorClientV11m(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V11M)


class YoloDetectorClientV11l(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V11L)


class YoloDetectorClientV12n(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V12N)


class YoloDetectorClientV12s(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V12S)


class YoloDetectorClientV12m(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V12M)


class YoloDetectorClientV12l(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V12L)
