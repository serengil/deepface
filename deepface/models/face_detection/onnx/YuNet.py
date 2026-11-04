# built-in dependencies
import os
from typing import Any, List, cast

# 3rd party dependencies
import cv2
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()

# pylint:disable=line-too-long
WEIGHTS_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# the graph is exported with a fixed (1, 3, 640, 640) shaped input
INPUT_SIZE = 640
STRIDES = [8, 16, 32]
NMS_THRESHOLD = 0.3
TOP_K = 5000


class YuNetClient(Detector):
    def __init__(self) -> None:
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a yunet detector model
        Returns:
            model (onnxruntime.InferenceSession)
        """
        return build_session(
            file_name="face_detection_yunet_2023mar.onnx",
            source_url=WEIGHTS_URL,
        )

    def detect_faces(self, img: NDArray[Any]) -> List[FacialAreaRegion]:
        """
        Detect and align face with yunet

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        # FaceDetector.detect_faces does not support score_threshold parameter.
        # We can set it via environment variable.
        score_threshold = float(os.environ.get("yunet_score_threshold", "0.9"))
        resp = []
        height, width = img.shape[0], img.shape[1]
        # resize image if it is too large, the graph is exported with a 640x640 input
        resized = False
        r: float = 1  # resize factor
        if height > 640 or width > 640:
            r = 640.0 / max(height, width)
            img = cv2.resize(img, (int(width * r), int(height * r)))
            height, width = img.shape[0], img.shape[1]
            resized = True
        faces = self.forward(img, score_threshold)
        for face in faces:
            # pylint: disable=W0105
            """
            The detection output faces is a two-dimension array of type CV_32F,
            whose rows are the detected face instances, columns are the location
            of a face and 5 facial landmarks.
            The format of each row is as follows:
            x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
            x_rcm, y_rcm, x_lcm, y_lcm,
            where x1, y1, w, h are the top-left coordinates, width and height of
            the face bounding box,
            {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye,
            left eye, nose tip, the right corner and left corner of the mouth respectively.
            """
            (x, y, w, h, x_le, y_le, x_re, y_re) = list(map(int, face[:8]))

            # YuNet returns negative coordinates if it thinks part of the detected face
            # is outside the frame.
            x = max(x, 0)
            y = max(y, 0)
            if resized:
                x, y, w, h = int(x / r), int(y / r), int(w / r), int(h / r)
                x_re, y_re, x_le, y_le = (
                    int(x_re / r),
                    int(y_re / r),
                    int(x_le / r),
                    int(y_le / r),
                )
            confidence = float(face[-1])

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=confidence,
                left_eye=(x_re, y_re),
                right_eye=(x_le, y_le),
            )
            resp.append(facial_area)
        return resp

    def forward(self, img: NDArray[Any], score_threshold: float) -> NDArray[Any]:
        """
        Run the graph and decode its outputs as opencv's FaceDetectorYN.detect does
        Args:
            img (np.ndarray): pre-loaded image in BGR, its sides must not exceed 640
            score_threshold (float): minimum score of the detections to keep
        Returns:
            faces (np.ndarray): (n, 15) shaped detections, each row is
                x1, y1, w, h, 5 facial landmarks and score
        """
        height, width = img.shape[0], img.shape[1]

        # zero pad bottom and right sides to fit the input size of the graph
        padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
        padded[:height, :width] = img
        # channels last to channels first, opencv's blobFromImage keeps BGR without scaling
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[np.newaxis])

        input_name = self.model.get_inputs()[0].name
        outputs = dict(
            zip(
                [output.name for output in self.model.get_outputs()],
                self.model.run(None, {input_name: blob}),
            )
        )

        boxes, landmarks, scores = [], [], []
        for stride in STRIDES:
            cols = INPUT_SIZE // stride
            cls_scores = np.clip(outputs[f"cls_{stride}"][0, :, 0], 0, 1)
            obj_scores = np.clip(outputs[f"obj_{stride}"][0, :, 0], 0, 1)
            bbox = outputs[f"bbox_{stride}"][0]
            kps = outputs[f"kps_{stride}"][0]

            score = np.sqrt(cls_scores * obj_scores)
            idx = np.where(score >= score_threshold)[0]
            rows, cols_idx = idx // cols, idx % cols

            cx = (cols_idx + bbox[idx, 0]) * stride
            cy = (rows + bbox[idx, 1]) * stride
            w = np.exp(bbox[idx, 2]) * stride
            h = np.exp(bbox[idx, 3]) * stride
            boxes.append(np.stack([cx - w / 2, cy - h / 2, w, h], axis=1))

            lms = kps[idx].reshape(-1, 5, 2)
            lms_x = (lms[:, :, 0] + cols_idx[:, np.newaxis]) * stride
            lms_y = (lms[:, :, 1] + rows[:, np.newaxis]) * stride
            landmarks.append(np.stack([lms_x, lms_y], axis=2).reshape(-1, 10))
            scores.append(score[idx])

        boxes_np = np.concatenate(boxes, axis=0)
        scores_np = np.concatenate(scores, axis=0)
        if len(scores_np) == 0:
            return np.empty((0, 15), dtype=np.float32)

        # pylint: disable=c-extension-no-member
        keep = np.array(
            cv2.dnn.NMSBoxes(
                boxes_np.tolist(),
                scores_np.tolist(),
                score_threshold,
                NMS_THRESHOLD,
                top_k=TOP_K,
            )
        ).flatten()
        return cast(
            NDArray[Any],
            np.concatenate(
                [
                    boxes_np[keep],
                    np.concatenate(landmarks, axis=0)[keep],
                    scores_np[keep, np.newaxis],
                ],
                axis=1,
            ).astype(np.float32),
        )
