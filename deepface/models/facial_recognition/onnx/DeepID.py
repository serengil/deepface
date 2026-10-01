# built-in dependencies
from typing import Any, List, Union

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_weights.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/deepid_weights.onnx",
]


# pylint: disable=too-few-public-methods
class DeepIdClient(OnnxFacialRecognition):
    """
    DeepId model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "DeepId"
        self.input_shape = (47, 55)
        self.output_shape = 160
        self.model = load_model()


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download DeepId's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="deepid_weights.onnx", source_url=url)
