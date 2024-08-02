import os
import logging
from datetime import datetime

# pylint: disable=broad-except
class Logger:
    """
    A Logger class for logging messages with a specific log level.

    The class follows the singleton design pattern, ensuring that only one
    instance of the Logger is created. The parameters of the first instance 
    are preserved across all instances.
    """

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(Logger, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        if not hasattr(self, "_singleton_initialized"):
            self._singleton_initialized = True # to prevent multiple initializations
            log_level = os.environ.get("DEEPFACE_LOG_LEVEL", str(logging.INFO))
            try:
                self.log_level = int(log_level)
            except Exception as err:
                self.dump_log(
                    f"Exception while parsing $DEEPFACE_LOG_LEVEL."
                    f"Expected int but it is {log_level} ({str(err)})."
                    "Setting app log level to info."
                )
                self.log_level = logging.INFO

    def info(self, message):
        if self.log_level <= logging.INFO:
            self.dump_log(f"{message}")

    def debug(self, message):
        if self.log_level <= logging.DEBUG:
            self.dump_log(f"🕷️ {message}")

    def warn(self, message):
        if self.log_level <= logging.WARNING:
            self.dump_log(f"⚠️ {message}")

    def error(self, message):
        if self.log_level <= logging.ERROR:
            self.dump_log(f"🔴 {message}")

    def critical(self, message):
        if self.log_level <= logging.CRITICAL:
            self.dump_log(f"💥 {message}")

    def dump_log(self, message):
        print(f"{str(datetime.now())[2:-7]} - {message}")
