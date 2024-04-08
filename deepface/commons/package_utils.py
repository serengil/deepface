# built-in dependencies
import os
import hashlib

# 3rd party dependencies
import tensorflow as tf

# package dependencies
from deepface.commons import logger as log

logger = log.get_singletonish_logger()


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


def find_hash_of_file(file_path: str) -> str:
    """
    Find the hash of given image file with its properties
        finding the hash of image content is costly operation
    Args:
        file_path (str): exact image path
    Returns:
        hash (str): digest with sha1 algorithm
    """
    file_stats = os.stat(file_path)

    # some properties
    file_size = file_stats.st_size
    creation_time = file_stats.st_ctime
    modification_time = file_stats.st_mtime

    properties = f"{file_size}-{creation_time}-{modification_time}"

    hasher = hashlib.sha1()
    hasher.update(properties.encode("utf-8"))
    return hasher.hexdigest()


def validate_for_keras3():
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
