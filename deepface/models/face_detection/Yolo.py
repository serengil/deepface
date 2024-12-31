# built-in dependencies
import os
from typing import List, Any
from enum import Enum

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger
from deepface.commons import weight_utils

logger = Logger()


class YoloModel(Enum):
    V8N = 0
    V11N = 1
    V11S = 2
    V11M = 3


# Model's weights paths
WEIGHT_NAMES = ["yolov8n-face.pt",
                "yolov11n-face.pt",
                "yolov11s-face.pt",
                "yolov11m-face.pt"]

# Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
WEIGHT_URLS = ["https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11s-face.pt",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt"]


class YoloDetectorClient(Detector):
    def __init__(self, model: YoloModel):
        super().__init__()
        self.model = self.build_model(model)

    def build_model(self, model: YoloModel) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """

        # Import the optional Ultralytics YOLO model
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector, ensure the library is installed. "
                "Please install using 'pip install ultralytics'"
            ) from e

        weight_file = weight_utils.download_weights_if_necessary(
            file_name=WEIGHT_NAMES[model.value], source_url=WEIGHT_URLS[model.value]
        )

        # Return face_detector
        return YOLO(weight_file)

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with yolo

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        # Detect faces
        results = self.model.predict(
            img,
            verbose=False,
            show=False,
            conf=float(os.getenv("YOLO_MIN_DETECTION_CONFIDENCE", "0.25")),
        )[0]

        # For each face, extract the bounding box, the landmarks and confidence
        for result in results:

            if result.boxes is None:
                continue

            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            right_eye = None
            left_eye = None

            # yolo-facev8 is detecting eyes through keypoints,
            # while for v11 keypoints are always None
            if result.keypoints is not None:
                # right_eye_conf = result.keypoints.conf[0][0]
                # left_eye_conf = result.keypoints.conf[0][1]
                right_eye = result.keypoints.xy[0][0].tolist()
                left_eye = result.keypoints.xy[0][1].tolist()

                # eyes are list of float, need to cast them tuple of int
                left_eye = tuple(int(i) for i in left_eye)
                right_eye = tuple(int(i) for i in right_eye)

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            resp.append(facial_area)

        return resp


class YoloDetectorClientV8n(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V8N)


class YoloDetectorClientV11n(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11N)


class YoloDetectorClientV11s(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11S)


class YoloDetectorClientV11m(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11M)
