import os
from deepface.commons.logger import Logger

logger = Logger()


def initialize_folder() -> None:
    """
    Initialize the folder for storing model weights.

    Raises:
        OSError: if the folder cannot be created.
    """
    home = get_deepface_home()
    deepface_home_path = os.path.join(home, ".deepface")
    weights_path = os.path.join(deepface_home_path, "weights")

    if not os.path.exists(deepface_home_path):
        os.makedirs(deepface_home_path, exist_ok=True)
        logger.info(f"Directory {deepface_home_path} has been created")

    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)
        logger.info(f"Directory {weights_path} has been created")


def get_deepface_home() -> str:
    """
    Get the home directory for storing model weights

    Returns:
        str: the home directory.
    """
    return str(os.getenv("DEEPFACE_HOME", default=os.path.expanduser("~")))
