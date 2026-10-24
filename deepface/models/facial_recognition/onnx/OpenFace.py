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
        logger.warn(
            "Please note that you built an underperforming OpenFace model - even in its best "
            "configuration, it scores 78.7% accuracy on LFW whereas human beings score 97.5%. "
            "It is not recommended for production purposes. "
            "Consider a better-performing model instead - see "
            "https://github.com/serengil/deepface/tree/master/benchmarks",
            show_once=True,
        )
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
