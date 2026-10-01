# built-in dependencies
from typing import Any, List, Union

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/arcface_weights.onnx",
]


# pylint: disable=too-few-public-methods
class ArcFaceClient(OnnxFacialRecognition):
    """
    ArcFace model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "ArcFace"
        self.input_shape = (112, 112)
        self.output_shape = 512
        self.model = load_model()


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download ArcFace's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="arcface_weights.onnx", source_url=url)
