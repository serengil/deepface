# built-in dependencies
from typing import Any, List, Union, cast

# 3rd party dependencies
from numpy.typing import NDArray

# project dependencies
from deepface.modules import verification
from deepface.commons.onnx_utils import build_session
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/vgg_face_weights.onnx",
]


# pylint: disable=too-few-public-methods
class VggFaceClient(OnnxFacialRecognition):
    """
    VGG-Face model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "VGG-Face"
        self.input_shape = (224, 224)
        self.output_shape = 4096
        self.model = load_model()

    def forward(self, img: NDArray[Any]) -> List[float]:
        """
        Generates embeddings using the VGG-Face model.
            This method incorporates an additional normalization layer.

        Args:
            img (np.ndarray): pre-loaded image in BGR with (224, 224, 3)
                or (n, 224, 224, 3) shape
        Returns
            embeddings (list): multi-dimensional vector
        """
        embeddings = self.predict(img)

        if embeddings.shape[0] == 1:
            embedding_norm = verification.l2_normalize(embeddings[0].tolist())
        else:
            embedding_norm = verification.l2_normalize(embeddings.tolist(), axis=1)

        return cast(List[float], embedding_norm.tolist())


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download VGG-Face's onnx graph if necessary and load it. The graph returns 4096
    dimensional descriptors, its 2622 way classifier was left out while exporting.
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="vgg_face_weights.onnx", source_url=url)
