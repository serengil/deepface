# built-in dependencies
import os
from typing import Any, List, Union, cast

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons import weight_utils
from deepface.commons.logger import Logger

logger = Logger()

# onnxruntime is an optional dependency, so it is imported only when a session is built.

# graphs exported from the pytorch models take channels last - (n, X, X, c) shaped - inputs
# as the tensorflow backend does. So, neither torch nor tensorflow is required to run them.

# set this environment variable to run onnx graphs on other execution providers, in the
# order of precedence, e.g. DEEPFACE_ONNX_PROVIDERS=coreml or DEEPFACE_ONNX_PROVIDERS=tensorrt,cuda
PROVIDERS_ENV_VAR = "DEEPFACE_ONNX_PROVIDERS"

# runs on cpu unless onnxruntime-gpu is installed
DEFAULT_PROVIDERS = "cuda,cpu"

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


def build_session(file_name: str, source_url: Union[str, List[str]]) -> Any:
    """
    Download an onnx graph if necessary and load it into an inference session
    Args:
        file_name (str): name of the onnx file under the weights folder
        source_url (str or list): url(s) to download the onnx file from
    Returns:
        session (onnxruntime.InferenceSession)
    """
    # onnxruntime is an optional dependency, enforce it to be installed if a model is built
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as err:
        raise ImportError(
            "onnxruntime is an optional dependency, ensure the library is installed. "
            "Please install using 'pip install onnxruntime' or "
            "'pip install onnxruntime-gpu' to run it on a gpu."
        ) from err

    weight_file = weight_utils.download_weights_if_necessary(
        file_name=file_name, source_url=source_url
    )

    providers = resolve_providers(ort.get_available_providers())

    # only log errors, some graphs (e.g. sface) flood stderr with harmless optimization warnings
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3

    try:
        return ort.InferenceSession(weight_file, sess_options=session_options, providers=providers)
    except Exception as err:
        raise ValueError(
            f"An exception occurred while loading the onnx graph from {weight_file}."
            "This might have happened due to an interruption during the download."
            "You may want to delete it and allow DeepFace to download it again during the next run."
            "If the issue persists, consider downloading the file directly from the source "
            "and copying it to the target folder."
        ) from err


def resolve_providers(available: List[str]) -> List[str]:
    """
    Find the execution providers DEEPFACE_ONNX_PROVIDERS lists, skipping the ones
    the installed onnxruntime does not offer
    Args:
        available (list): providers the installed onnxruntime offers
    Returns:
        providers (list)
    """
    requested = os.environ.get(PROVIDERS_ENV_VAR) or DEFAULT_PROVIDERS
    names = requested.lower().replace(" ", "").split(",")

    providers = []
    for name in names:
        if name not in PROVIDERS:
            raise ValueError(f"{PROVIDERS_ENV_VAR} must list some of {list(PROVIDERS)}")
        if PROVIDERS[name] in available and PROVIDERS[name] not in providers:
            providers.append(PROVIDERS[name])

    # warn if the user explicitly asked for accelerators but none of them is installed,
    # otherwise graphs silently run on cpu
    accelerators = [PROVIDERS[name] for name in names if name != "cpu"]
    if (
        os.environ.get(PROVIDERS_ENV_VAR)
        and accelerators
        and not any(provider in providers for provider in accelerators)
    ):
        logger.warn(
            f"None of {accelerators} set in {PROVIDERS_ENV_VAR} is offered by the installed "
            f"onnxruntime, which offers {available}. Falling back to cpu. You may need to "
            "install the matching package, e.g. onnxruntime-gpu for cuda."
        )

    # cpu is the last resort for the operations other providers do not support
    if PROVIDERS["cpu"] not in providers:
        providers.append(PROVIDERS["cpu"])

    logger.debug(f"onnx graphs run on {providers}")
    return providers


def run_session(session: Any, img: NDArray[Any]) -> NDArray[Any]:
    """
    Feed channels last images into an onnx inference session
    Args:
        session (onnxruntime.InferenceSession): session built with build_session
        img (np.ndarray): (X, X, c) or (n, X, X, c) shaped images
    Returns:
        outputs (np.ndarray): (n, ...) shaped outputs of the graph
    """
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    if img.ndim != 4 or img.shape[0] < 1:
        raise ValueError(f"Input image must be (n, X, X, c) shaped but it is {img.shape}")

    inputs = {session.get_inputs()[0].name: np.ascontiguousarray(img, dtype=np.float32)}
    return cast(NDArray[Any], session.run(None, inputs)[0])
