# project dependencies
from deepface.api.src.dependencies.variables import Variables
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=too-few-public-methods
class AuthService:
    def __init__(self, variables: Variables) -> None:
        self.variables = variables
        self.is_auth_enabled = (
            self.variables.auth_token is not None and len(self.variables.auth_token) > 0
        )

    def validate_token(self, token: str) -> bool:
        """
        Validates the provided authentication token.

        Args:
            token (Optional[str]): The authentication token to validate.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        return token == self.variables.auth_token
