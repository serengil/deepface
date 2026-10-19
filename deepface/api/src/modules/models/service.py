# built-in dependencies
from typing import Optional

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=too-few-public-methods
class ModelService:
    def __init__(
        self,
        face_recognition_models: Optional[str] = None,
        face_detection_models: Optional[str] = None,
    ) -> None:
        self.face_recognition_models = face_recognition_models
        self.face_detection_models = face_detection_models

    def load_models_on_startup(self) -> None:
        """Load models on startup to reduce latency on first request."""
        if self.face_recognition_models is not None:
            for model in self.face_recognition_models.split(","):
                DeepFace.build_model(task="facial_recognition", model_name=model.strip())
                logger.info(f"Facial Recognition Model {model} loaded on startup.")
        if self.face_detection_models is not None:
            for model in self.face_detection_models.split(","):
                DeepFace.build_model(task="face_detector", model_name=model.strip())
                logger.info(f"Face Detector Model {model} loaded on startup.")
