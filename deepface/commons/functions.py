import os
from pathlib import Path

# 3rd party dependencies
import tensorflow as tf

# package dependencies
from deepface.commons.logger import Logger

logger = Logger(module="commons.functions")


def get_tf_major_version() -> int:
    return int(tf.__version__.split(".", maxsplit=1)[0])


def initialize_folder() -> None:
    """Initialize the folder for storing weights and models.

    Raises:
        OSError: if the folder cannot be created.
    """
    home = get_deepface_home()
    deepFaceHomePath = home + "/.deepface"
    weightsPath = deepFaceHomePath + "/weights"

    if not os.path.exists(deepFaceHomePath):
        os.makedirs(deepFaceHomePath, exist_ok=True)
        logger.info(f"Directory {home}/.deepface created")

    if not os.path.exists(weightsPath):
        os.makedirs(weightsPath, exist_ok=True)
        logger.info(f"Directory {home}/.deepface/weights created")


def get_deepface_home() -> str:
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory.
    """
    return str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))
