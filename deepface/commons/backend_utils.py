# built-in dependencies
import os
import importlib.util
from typing import Optional

# package dependencies
from deepface.commons.logger import Logger

logger = Logger()

# deepface can run on tensorflow, pytorch or onnxruntime. this module decides which one of
# them will be used, and it must not import any of them - importing a framework here
# would force every user to have it installed.

TENSORFLOW = "tensorflow"
PYTORCH = "pytorch"
ONNX = "onnx"

# backend engines in the order of precedence when none of them is enforced
BACKENDS = [TENSORFLOW, PYTORCH, ONNX]

# set this environment variable to enforce a backend engine
BACKEND_ENGINE_ENV_VAR = "DEEPFACE_BACKEND_ENGINE"

# packages a backend engine requires. keras is not listed for tensorflow because it
# comes with it as tensorflow.keras in tf2, and that is what deepface imports there.
# standalone keras is required for tf1 only, and it is installed with the extra anyway.
BACKEND_REQUIREMENTS = {
    TENSORFLOW: ["tensorflow"],
    PYTORCH: ["torch"],
    ONNX: ["onnxruntime"],
}

_backend_engine: Optional[str] = None


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed without importing it
    Args:
        package_name (str): name of the package
    Returns:
        installed (bool)
    """
    try:
        return importlib.util.find_spec(package_name) is not None
    except (ImportError, ValueError):
        # a broken installation may raise while its spec is being looked up
        return False


def is_backend_available(backend: str) -> bool:
    """
    Check if every requirement of a backend engine is installed
    Args:
        backend (str): tensorflow, pytorch or onnx
    Returns:
        available (bool)
    """
    return all(is_package_installed(package) for package in BACKEND_REQUIREMENTS[backend])


def get_backend_engine() -> str:
    """
    Find the backend engine deepface will run on. It is set once and then cached.

    The engine is the one the DEEPFACE_BACKEND_ENGINE environment variable enforces,
    or tensorflow when it is installed, or pytorch when it is installed, or onnx when
    onnxruntime is installed.
    Returns:
        backend (str): tensorflow, pytorch or onnx
    """
    global _backend_engine  # pylint: disable=global-statement

    if _backend_engine is not None:
        return _backend_engine

    enforced = os.environ.get(BACKEND_ENGINE_ENV_VAR)
    if enforced:
        backend = enforced.strip().lower()
        if backend not in BACKENDS:
            raise ValueError(
                f"Unimplemented backend engine - {enforced}. "
                f"{BACKEND_ENGINE_ENV_VAR} must be one of {BACKENDS}."
            )
        if not is_backend_available(backend):
            packages = BACKEND_REQUIREMENTS[backend]
            requirements = " and ".join(packages)
            verb = "is" if len(packages) == 1 else "are"
            # onnxruntime is a base requirement, there is no extra to install it
            command = (
                "pip install onnxruntime" if backend == ONNX else f"pip install deepface[{backend}]"
            )
            raise ValueError(
                f"{BACKEND_ENGINE_ENV_VAR} is set to {backend} but {requirements} "
                f"{verb} not installed. Please run `{command}` "
                f"or unset {BACKEND_ENGINE_ENV_VAR}."
            )
        logger.debug(f"{backend} backend engine is enforced with {BACKEND_ENGINE_ENV_VAR}")
        _backend_engine = backend
        return _backend_engine

    for backend in BACKENDS:
        if is_backend_available(backend):
            logger.debug(f"{backend} backend engine will be used")
            _backend_engine = backend
            return _backend_engine

    raise ValueError(
        "deepface requires tensorflow, pytorch or onnxruntime to be installed, but none of "
        "them is available. onnxruntime comes with deepface, please run `pip install onnxruntime` "
        "to restore it, or `pip install deepface[tensorflow]` or `pip install deepface[pytorch]` "
        "to install the backend engine you want to use."
    )


def is_tensorflow() -> bool:
    """
    Check if deepface runs on tensorflow
    Returns:
        result (bool)
    """
    return get_backend_engine() == TENSORFLOW


def is_pytorch() -> bool:
    """
    Check if deepface runs on pytorch
    Returns:
        result (bool)
    """
    return get_backend_engine() == PYTORCH


def is_onnx() -> bool:
    """
    Check if deepface runs on onnxruntime
    Returns:
        result (bool)
    """
    return get_backend_engine() == ONNX
