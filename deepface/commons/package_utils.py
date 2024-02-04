import json

# 3rd party dependencies
import tensorflow as tf

# package dependencies
from deepface.commons.logger import Logger
from deepface.commons import constant

logger = Logger(module="commons.package_utils")


def get_tf_major_version() -> int:
    """
    Find tensorflow's major version
    Returns
        major_version (int)
    """
    return int(tf.__version__.split(".", maxsplit=1)[0])


def find_package_version() -> str:
    """
    Find the currenct package version
    Returns:
        version (str)
    """
    version_info = "N/A"
    try:
        with open(f"{constant.ROOT_DIR}/package_info.json", "r", encoding="utf-8") as f:
            package_info = json.load(f)
        version_info = package_info["version"]
    except Exception as err:  # pylint: disable=broad-except, unused-variable
        pass

    return version_info
