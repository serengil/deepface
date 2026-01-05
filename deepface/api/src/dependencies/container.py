# project dependencies
from deepface.api.src.modules.auth.service import AuthService
from deepface.api.src.dependencies.variables import Variables


# pylint: disable=too-few-public-methods
class Container:
    def __init__(self, variables: Variables) -> None:
        # once you have variables, you can connect dbs and other services here
        self.auth_service = AuthService(auth_token=variables.auth_token)
