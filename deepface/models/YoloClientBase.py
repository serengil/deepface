# built-in dependencies
from typing import Any

# project dependencies
from deepface.models.YoloModel import YoloModel, WEIGHT_URLS, WEIGHT_NAMES
from deepface.commons import weight_utils
from deepface.commons.logger import Logger

logger = Logger()


class YoloClientBase:
    def __init__(self, model: YoloModel):
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
