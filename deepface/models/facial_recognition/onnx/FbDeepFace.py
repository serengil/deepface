# built-in dependencies
from typing import Any, List, Union

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/VGGFace2_DeepFace_weights_val-0.9034.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/VGGFace2_DeepFace_weights_val-0.9034.onnx",
]


# pylint: disable=too-few-public-methods
class DeepFaceClient(OnnxFacialRecognition):
    """
    Fb's DeepFace model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "DeepFace"
        self.input_shape = (152, 152)
        self.output_shape = 4096
        self.model = load_model()


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download Fb's DeepFace onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="VGGFace2_DeepFace_weights_val-0.9034.onnx", source_url=url)
