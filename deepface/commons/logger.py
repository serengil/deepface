import os
import hashlib
from typing import Any, Set
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

    def __new__(cls) -> "Logger":
        if cls.__instance is None:
            cls.__instance = super(Logger, cls).__new__(cls)
        return cls.__instance

    def __init__(self) -> None:
        if not hasattr(self, "_singleton_initialized"):
            self._singleton_initialized = True  # to prevent multiple initializations
            self._shown_warnings: Set[str] = set()  # hashes of show_once warnings
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

    def info(self, message: Any) -> None:
        """
        Logs an info message if the log level is set to INFO or lower.
        Args:
            message: The message to log.
        """
        if self.log_level <= logging.INFO:
            self.dump_log(f"{message}")

    def debug(self, message: Any) -> None:
        """
        Logs a debug message if the log level is set to DEBUG or lower.
        Args:
            message: The message to log.
        """
        if self.log_level <= logging.DEBUG:
            self.dump_log(f"🕷️ {message}")

    def warn(self, message: Any, show_once: bool = False) -> None:
        """
        Logs a warning message if the log level is set to WARNING or lower.
        Args:
            message: The message to log.
            show_once: If True, the same message is logged only once per process.
        """
        if show_once:
            message_hash = hashlib.sha256(str(message).encode("utf-8")).hexdigest()
            if message_hash in self._shown_warnings:
                return
            self._shown_warnings.add(message_hash)

        if self.log_level <= logging.WARNING:
            self.dump_log(f"⚠️ {message}")

    def error(self, message: Any) -> None:
        """
        Logs an error message if the log level is set to ERROR or lower.
        Args:
            message: The message to log.
        """
        if self.log_level <= logging.ERROR:
            self.dump_log(f"🔴 {message}")

    def critical(self, message: Any) -> None:
        """
        Logs a critical message if the log level is set to CRITICAL or lower.
        Args:
            message: The message to log.
        """
        if self.log_level <= logging.CRITICAL:
            self.dump_log(f"💥 {message}")

    def dump_log(self, message: Any) -> None:
        """
        Dumps the log message to the console.
        Args:
            message: The message to log.
        """
        print(f"{str(datetime.now())[2:-7]} - {message}")
