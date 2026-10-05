# built-in dependencies
import hashlib
import importlib.util
import logging
from typing import Optional

# package dependencies
from deepface.commons.logger import Logger

# tensorflow is imported within the functions below on purpose, importing it here would
# force it onto users running deepface on pytorch

logger = Logger()


def get_tf_major_version() -> int:
    """
    Find tensorflow's major version
    Returns
        major_version (int)
    """
    import tensorflow as tf

    return int(tf.__version__.split(".", maxsplit=1)[0])


def get_tf_minor_version() -> int:
    """
    Find tensorflow's minor version
    Returns
        minor_version (int)
    """
    import tensorflow as tf

    return int(tf.__version__.split(".", maxsplit=-1)[1])


def validate_for_keras3() -> None:
    """
    Ensure tf_keras is available when tensorflow needs it
    """
    import tensorflow as tf

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


def configure_tensorflow_logging() -> None:
    """
    Silence tensorflow, it is noisy with its own warnings
    """
    import tensorflow as tf

    if get_tf_major_version() == 2:
        tf.get_logger().setLevel(logging.ERROR)


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


def resolve_dataframe_return_type(return_type: Optional[str]) -> str:
    """
    Resolve the return type of find and search functions. pandas is an optional dependency,
        so results are returned as pandas dataframes by default only if it is installed.
    Args:
        return_type (str): 'pandas', 'dict' or None. If None, it is resolved to 'pandas'
            if pandas is installed, otherwise to 'dict'.
    Returns:
        return_type (str): 'pandas' or 'dict'
    """
    if return_type not in (None, "pandas", "dict"):
        raise ValueError(f"Unsupported return_type: {return_type}. Options: 'pandas', 'dict'.")

    # check pandas availability without importing it
    try:
        pandas_installed = importlib.util.find_spec("pandas") is not None
    except (ImportError, ValueError):
        pandas_installed = False

    if return_type is None:
        return "pandas" if pandas_installed else "dict"

    if return_type == "pandas" and pandas_installed is False:
        raise ImportError(
            "pandas is an optional dependency, it is required for return_type='pandas'. "
            "Please either install it using 'pip install pandas' or set return_type to 'dict'."
        )

    return return_type
