# built-in dependencies
import os


# pylint: disable=too-few-public-methods
class Variables:
    def __init__(self) -> None:
        self.database_type = os.getenv("DEEPFACE_DATABASE_TYPE", "postgres").lower()

        if self.database_type == "postgres":
            conection_details = os.getenv("DEEPFACE_POSTGRES_URI")
        elif self.database_type == "mongo":
            conection_details = os.getenv("DEEPFACE_MONGO_URI")
        elif self.database_type == "weaviate":
            conection_details = os.getenv("DEEPFACE_WEAVIATE_URI")
        else:
            conection_details = None

        self.conection_details = os.getenv("DEEPFACE_CONNECTION_DETAILS") or conection_details

        self.face_recognition_models = os.getenv("DEEPFACE_FACE_RECOGNITION_MODELS")
        self.face_detection_models = os.getenv("DEEPFACE_FACE_DETECTION_MODELS")

        self.auth_token = os.getenv("DEEPFACE_AUTH_TOKEN")
