# built-in dependencies
import os
from typing import List, Any, Tuple

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import cv2

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=c-extension-no-member

WEIGHTS_URL = "https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx"


class CenterFaceClient(Detector):
    def __init__(self) -> None:
        self.model = self.build_model()

    def build_model(self) -> "CenterFace":
        """
        Download pre-trained weights of CenterFace model if necessary and load built model
        """
        # the graph declares a fixed (10, 3, 32, 32) shaped input which onnxruntime enforces,
        # but it is fully convolutional, so its declared shapes are dropped
        return CenterFace(
            session=build_session(
                file_name="centerface.onnx", source_url=WEIGHTS_URL, dynamic_shapes=True
            )
        )

    def detect_faces(self, img: NDArray[Any]) -> List["FacialAreaRegion"]:
        """
        Detect and align face with CenterFace

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        threshold = float(os.getenv("CENTERFACE_THRESHOLD", "0.35"))

        detections, landmarks = self.model.forward(
            img, img.shape[0], img.shape[1], threshold=threshold
        )

        for i, detection in enumerate(detections):
            boxes, confidence = detection[:4], detection[4]

            x = boxes[0]
            y = boxes[1]
            w = boxes[2] - x
            h = boxes[3] - y

            landmark = landmarks[i]

            right_eye = (int(landmark[0]), int(landmark[1]))
            left_eye = (int(landmark[2]), int(landmark[3]))
            # nose = (int(landmark[4]), int(landmark [5]))
            # mouth_right = (int(landmark[6]), int(landmark [7]))
            # mouth_left = (int(landmark[8]), int(landmark [9]))

            facial_area = FacialAreaRegion(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=min(max(0, float(confidence)), 1.0),
            )
            resp.append(facial_area)

        return resp


class CenterFace:
    """
    This class is heavily inspired from
        github.com/Star-Clouds/CenterFace/blob/master/prj-python/centerface.py
    """

    def __init__(self, session: Any) -> None:
        self.session = session
        self.img_h_new: int = 0
        self.img_w_new: int = 0
        self.scale_h: float = 0
        self.scale_w: float = 0

    def forward(self, img: NDArray[Any], height: int, width: int, threshold: float = 0.5) -> Any:
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
        return self.inference_onnx(img, threshold)

    def inference_onnx(self, img: NDArray[Any], threshold: float) -> Any:
        resized = cv2.resize(img, (self.img_w_new, self.img_h_new))
        # BGR to RGB, channels last to channels first, no scaling as opencv's blobFromImage
        blob = np.ascontiguousarray(
            resized[..., ::-1].transpose(2, 0, 1)[np.newaxis], dtype=np.float32
        )
        input_name = self.session.get_inputs()[0].name
        heatmap, scale, offset, lms = self.session.run(
            ["537", "538", "539", "540"], {input_name: blob}
        )
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def transform(self, h: int, w: int) -> Tuple[int, int, float, float]:
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(
        self,
        heatmap: NDArray[Any],
        lms: NDArray[Any],
        offset: NDArray[Any],
        scale: NDArray[Any],
        threshold: float,
    ) -> Any:
        dets, lms = self.decode(
            heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold
        )
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = (
                dets[:, 0:4:2] / self.scale_w,
                dets[:, 1:4:2] / self.scale_h,
            )
            lms[:, 0:10:2], lms[:, 1:10:2] = (
                lms[:, 0:10:2] / self.scale_w,
                lms[:, 1:10:2] / self.scale_h,
            )
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            lms = np.empty(shape=[0, 10], dtype=np.float32)
        return dets, lms

    def decode(
        self,
        heatmap: NDArray[Any],
        scale: NDArray[Any],
        offset: NDArray[Any],
        landmark: NDArray[Any],
        size: Tuple[int, int],
        threshold: float = 0.1,
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if c0.size == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 10), dtype=np.float32)

        heights = np.exp(scale0[c0, c1]) * 4
        widths = np.exp(scale1[c0, c1]) * 4
        # Decode landmarks and both corners from the uncut prediction. Clipping
        # the origin first translates an edge face and its eyes into the image.
        left = (c1 + offset1[c0, c1] + 0.5) * 4 - widths / 2
        top = (c0 + offset0[c0, c1] + 0.5) * 4 - heights / 2
        boxes_np = np.column_stack(
            (
                np.clip(left, 0, size[1]),
                np.clip(top, 0, size[0]),
                np.clip(left + widths, 0, size[1]),
                np.clip(top + heights, 0, size[0]),
                heatmap[c0, c1],
            )
        ).astype(np.float32)
        keep = self.nms(boxes_np[:, :4], boxes_np[:, 4], 0.3)
        boxes_np = boxes_np[keep]
        # Materialize the ten landmark coordinates only for surviving proposals.
        rows, cols = c0[keep], c1[keep]
        lms_np: NDArray[Any] = np.empty((len(keep), 10), dtype=np.float32)
        lms_np[:, 0::2] = (
            landmark[0, 1::2, :, :][:, rows, cols].T * widths[keep, None] + left[keep, None]
        )
        lms_np[:, 1::2] = (
            landmark[0, 0::2, :, :][:, rows, cols].T * heights[keep, None] + top[keep, None]
        )
        return boxes_np, lms_np

    def nms(self, boxes: NDArray[Any], scores: NDArray[Any], nms_thresh: float) -> List[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            rest = order[1:]
            w = np.maximum(0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
            h = np.maximum(0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
            inter = w * h
            overlap = inter / (areas[i] + areas[rest] - inter)
            order = rest[overlap < nms_thresh]

        return keep
