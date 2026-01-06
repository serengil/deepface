# built-in dependencies
from typing import Optional, Dict, Any

# project dependencies
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=too-few-public-methods
class AuthService:
    def __init__(self, auth_token: Optional[str] = None) -> None:
        self.auth_token = auth_token
        self.is_auth_enabled = auth_token is not None and len(auth_token) > 0

    def extract_token(self, auth_header: Optional[str]) -> Optional[str]:
        if not auth_header:
            return None
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        return None

    def validate(self, headers: Dict[str, Any]) -> bool:
        if not self.is_auth_enabled:
            logger.debug("Authentication is disabled. Skipping token validation.")
            return True

        token = self.extract_token(headers.get("Authorization"))
        if not token:
            logger.debug("No authentication token provided. Validation failed.")
            return False

        if token != self.auth_token:
            logger.debug("Invalid authentication token provided. Validation failed.")
            return False

        logger.debug("Authentication token validated successfully.")
        return True
