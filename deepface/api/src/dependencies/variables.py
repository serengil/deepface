# built-in dependencies
import os

# project dependencies
from deepface.modules.database.inventory import database_inventory


# pylint: disable=too-few-public-methods
class Variables:
    def __init__(self) -> None:
        self.database_type = os.getenv("DEEPFACE_DATABASE_TYPE", "postgres").lower()

        if database_inventory.get(self.database_type) is None:
            raise ValueError(f"Unsupported database type: {self.database_type}")

        connection_string = database_inventory[self.database_type]["connection_string"]
        conection_details = os.getenv(connection_string)
        self.conection_details = os.getenv("DEEPFACE_CONNECTION_DETAILS") or conection_details

        self.face_recognition_models = os.getenv("DEEPFACE_FACE_RECOGNITION_MODELS")
        self.face_detection_models = os.getenv("DEEPFACE_FACE_DETECTION_MODELS")

        self.auth_token = os.getenv("DEEPFACE_AUTH_TOKEN")
