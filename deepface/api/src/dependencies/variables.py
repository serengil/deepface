# built-in dependencies
import os


# pylint: disable=too-few-public-methods
class Variables:
    def __init__(self) -> None:
        self.database_type = os.getenv("DEEPFACE_DATABASE_TYPE", "postgres").lower()
        self.conection_details = os.getenv("DEEPFACE_CONNECTION_DETAILS")
        self.face_recognition_models = os.getenv("DEEPFACE_FACE_RECOGNITION_MODELS")
        self.face_detection_models = os.getenv("DEEPFACE_FACE_DETECTION_MODELS")
