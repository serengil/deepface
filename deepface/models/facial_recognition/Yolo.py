# built-in dependencies
from typing import List, Any
from enum import Enum

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.models.FacialRecognition import FacialRecognition
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


class YoloFacialRecognitionClient(FacialRecognition):
    def __init__(self, model: YoloModel):
        super().__init__(model)
        self.model_name = "Yolo"
        self.input_shape = None
        self.output_shape = 512

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

    def forward(self, img: np.ndarray) -> List[float]:
        return self.model.embed(img)[0].tolist()


class YoloFacialRecognitionClientV8n(YoloFacialRecognitionClient):
    def __init__(self):
        super().__init__(YoloModel.V8N)


class YoloFacialRecognitionClientV11n(YoloFacialRecognitionClient):
    def __init__(self):
        super().__init__(YoloModel.V11N)


class YoloFacialRecognitionClientV11s(YoloFacialRecognitionClient):
    def __init__(self):
        super().__init__(YoloModel.V11S)


class YoloFacialRecognitionClientV11m(YoloFacialRecognitionClient):
    def __init__(self):
        super().__init__(YoloModel.V11M)
