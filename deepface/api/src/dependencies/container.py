# project dependencies
from deepface.api.src.modules.auth.service import AuthService
from deepface.api.src.modules.models.service import ModelService
from deepface.api.src.dependencies.variables import Variables


# pylint: disable=too-few-public-methods
class Container:
    def __init__(self, variables: Variables) -> None:
        # once you have variables, you can connect dbs and other services here
        self.auth_service = AuthService(auth_token=variables.auth_token)
        self.model_service = ModelService(
            face_recognition_models=variables.face_recognition_models,
            face_detection_models=variables.face_detection_models,
        )
