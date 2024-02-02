import os
from pathlib import Path
from deepface.commons.logger import Logger

logger = Logger(module="deepface/commons/folder_utils.py")


def initialize_folder() -> None:
    """
    Initialize the folder for storing model weights.

    Raises:
        OSError: if the folder cannot be created.
    """
    home = get_deepface_home()
    deepface_home_path = home + "/.deepface"
    weights_path = deepface_home_path + "/weights"

    if not os.path.exists(deepface_home_path):
        os.makedirs(deepface_home_path, exist_ok=True)
        logger.info(f"Directory {home}/.deepface created")

    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)
        logger.info(f"Directory {home}/.deepface/weights created")


def get_deepface_home() -> str:
    """
    Get the home directory for storing model weights

    Returns:
        str: the home directory.
    """
    return str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))
