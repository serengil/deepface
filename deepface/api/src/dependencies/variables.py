# built-in dependencies
import os


# pylint: disable=too-few-public-methods
class Variables:
    def __init__(self) -> None:
        self.database_type = os.getenv("DATABASE_TYPE", "postgres").lower()
        self.conection_details = os.getenv("DEEPFACE_CONNECTION_DETAILS")
