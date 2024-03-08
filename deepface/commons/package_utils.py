# built-in dependencies
import hashlib

# 3rd party dependencies
import tensorflow as tf

# package dependencies
from deepface.commons.logger import Logger

logger = Logger(module="commons.package_utils")


def get_tf_major_version() -> int:
    """
    Find tensorflow's major version
    Returns
        major_version (int)
    """
    return int(tf.__version__.split(".", maxsplit=1)[0])


def find_hash_of_file(file_path: str) -> str:
    """
    Find hash of image file
    Args:
        file_path (str): exact image path
    Returns:
        hash (str): digest with sha1 algorithm
    """
    with open(file_path, "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()
    return digest
