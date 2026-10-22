# built-in dependencies
import os
from typing import Any, cast

# third-party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons import folder_utils
from deepface.commons.onnx_utils import build_session
from deepface.commons.logger import Logger
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.modules.exceptions import InvalidEmbeddingsShapeError

logger = Logger()

# pylint: disable=line-too-long, too-few-public-methods
WEIGHTS_URL = (
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg"
)

# insightface's ArcFaceONNX.get_feat feeds images as (x - 127.5) / 127.5 scaled RGB
INPUT_MEAN = 127.5
INPUT_STD = 127.5


class Buffalo_L(OnnxFacialRecognition):
    """
    InsightFace's Buffalo_L recognition model (webface_r50) - onnx backend
    """

    def __init__(self) -> None:
        self.model = load_model()
        self.model_name = "Buffalo_L"
        self.input_shape = (112, 112)
        self.output_shape = 512

    def preprocess(self, img: NDArray[Any]) -> NDArray[Any]:
        """
        Preprocess the input image or batch of images as insightface's ArcFaceONNX.get_feat does.

        Args:
            img: BGR input image or batch in [0, 1] scale with shape
            (112, 112, 3) or (batch_size, 112, 112, 3).

        Returns:
            (batch_size, 3, 112, 112) shaped RGB blob in [-1, 1] scale.
        """
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)  # Convert single image to batch of 1
        elif img.ndim != 4:
            raise InvalidEmbeddingsShapeError(
                f"Input must be (112, 112, 3) or (X, 112, 112, 3). Got {img.shape}"
            )

        # BGR to RGB, channels last to channels first
        blob = (img * 255 - INPUT_MEAN) / INPUT_STD
        return np.ascontiguousarray(blob[..., ::-1].transpose(0, 3, 1, 2), dtype=np.float32)

    def predict(self, img: NDArray[Any]) -> NDArray[Any]:
        """
        Find raw representations of given images
        Args:
            img (np.ndarray): pre-loaded image in BGR with (112, 112, 3) or (n, 112, 112, 3)
                shape and [0, 1] scale
        Returns:
            embeddings (np.ndarray): (n, 512) shaped representations
        """
        blob = self.preprocess(img)
        model_input = self.model.get_inputs()[0]

        # feed images one by one if the graph is exported with a fixed batch size
        if isinstance(model_input.shape[0], int):
            embeddings = [
                self.model.run(None, {model_input.name: blob[i : i + 1]})[0]
                for i in range(blob.shape[0])
            ]
            return cast(NDArray[Any], np.concatenate(embeddings, axis=0))

        return cast(NDArray[Any], self.model.run(None, {model_input.name: blob})[0])


def load_model(url: str = WEIGHTS_URL) -> Any:
    """
    Download Buffalo_L's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    # the graph is stored under a sub folder of the weights folder
    buffalo_l_dir = os.path.join(
        folder_utils.get_deepface_home(), ".deepface", "weights", "buffalo_l"
    )
    os.makedirs(buffalo_l_dir, exist_ok=True)

    return build_session(file_name=os.path.join("buffalo_l", "webface_r50.onnx"), source_url=url)
