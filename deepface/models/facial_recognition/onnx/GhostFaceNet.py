# built-in dependencies
from typing import Any, List, Union

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/ghostfacenet_v1.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/ghostfacenet_v1.onnx",
]


# pylint: disable=too-few-public-methods
class GhostFaceNetClient(OnnxFacialRecognition):
    """
    GhostFaceNet model class - onnx backend
    Repo: https://github.com/HamadYA/GhostFaceNets
    Pre-trained weights: https://github.com/HamadYA/GhostFaceNets/releases/tag/v1.2
    """

    def __init__(self) -> None:
        self.model_name = "GhostFaceNet"
        self.input_shape = (112, 112)
        self.output_shape = 512
        self.model = load_model()


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download GhostFaceNet's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="ghostfacenet_v1.onnx", source_url=url)
