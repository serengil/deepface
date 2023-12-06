import os
import logging

# pylint: disable=broad-except
class Logger:
    def __init__(self, module=None):
        self.module = module
        log_level = os.environ.get("DEEPFACE_LOG_LEVEL", str(logging.INFO))
        try:
            self.log_level = int(log_level)
        except Exception as err:
            self.dump_log(
                f"Exception while parsing $DEEPFACE_LOG_LEVEL."
                f"Expected int but it is {log_level} ({str(err)})"
            )
            self.log_level = logging.INFO

    def info(self, message):
        if self.log_level <= logging.INFO:
            self.dump_log(message)

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
        print(message)
