# built-in dependencies
from typing import List

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.models.YoloClientBase import YoloClientBase
from deepface.models.YoloModel import YoloModel
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()


class YoloFacialRecognitionClient(YoloClientBase, FacialRecognition):
    def __init__(self, model: YoloModel):
        super().__init__(model)
        self.model_name = "Yolo"
        self.input_shape = None
        self.output_shape = 512

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
