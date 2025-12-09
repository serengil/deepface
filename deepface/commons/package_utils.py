# built-in dependencies
import hashlib

# 3rd party dependencies
import tensorflow as tf

# package dependencies
from deepface.commons.logger import Logger

logger = Logger()


def get_tf_major_version() -> int:
    """
    Find tensorflow's major version
    Returns
        major_version (int)
    """
    return int(tf.__version__.split(".", maxsplit=1)[0])


def get_tf_minor_version() -> int:
    """
    Find tensorflow's minor version
    Returns
        minor_version (int)
    """
    return int(tf.__version__.split(".", maxsplit=-1)[1])


def validate_for_keras3() -> None:
    tf_major = get_tf_major_version()
    tf_minor = get_tf_minor_version()

    # tf_keras is a must dependency after tf 2.16
    if tf_major == 1 or (tf_major == 2 and tf_minor < 16):
        return

    try:
        import tf_keras

        logger.debug(f"tf_keras is already available - {tf_keras.__version__}")
    except ImportError as err:
        # you may consider to install that package here
        raise ValueError(
            f"You have tensorflow {tf.__version__} and this requires "
            "tf-keras package. Please run `pip install tf-keras` "
            "or downgrade your tensorflow."
        ) from err


def find_file_hash(file_path: str, hash_algorithm: str = "sha256") -> str:
    """
    Find the hash of a given file with its content
    Args:
        file_path (str): exact path of a given file
        hash_algorithm (str): hash algorithm
    Returns:
        hash (str)
    """
    hash_func = hashlib.new(hash_algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()
