# built-in dependencies
from typing import Any, List, Union

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/openface_weights.onnx",
]


# pylint: disable=too-few-public-methods
class OpenFaceClient(OnnxFacialRecognition):
    """
    OpenFace model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "OpenFace"
        self.input_shape = (96, 96)
        self.output_shape = 128
        self.model = load_model()


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download OpenFace's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="openface_weights.onnx", source_url=url)
