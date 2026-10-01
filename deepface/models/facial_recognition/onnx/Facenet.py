# built-in dependencies
from typing import Any, List, Union

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

FACENET128_WEIGHTS = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/facenet_weights.onnx",
]
FACENET512_WEIGHTS = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/facenet512_weights.onnx",
]


# pylint: disable=too-few-public-methods
class FaceNet128dClient(OnnxFacialRecognition):
    """
    FaceNet-128d model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "FaceNet-128d"
        self.input_shape = (160, 160)
        self.output_shape = 128
        self.model = load_facenet128d_model()


class FaceNet512dClient(OnnxFacialRecognition):
    """
    FaceNet-512d model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "FaceNet-512d"
        self.input_shape = (160, 160)
        self.output_shape = 512
        self.model = load_facenet512d_model()


def load_facenet128d_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = FACENET128_WEIGHTS,
) -> Any:
    """
    Download FaceNet-128d onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="facenet_weights.onnx", source_url=url)


def load_facenet512d_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = FACENET512_WEIGHTS,
) -> Any:
    """
    Download FaceNet-512d onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="facenet512_weights.onnx", source_url=url)
