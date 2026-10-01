# built-in dependencies
from typing import Any, cast

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.facial_recognition.onnx.OnnxFacialRecognition import OnnxFacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long, too-few-public-methods
WEIGHTS_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


class SFaceClient(OnnxFacialRecognition):
    """
    SFace model class - onnx backend
    """

    def __init__(self) -> None:
        self.model = load_model()
        self.model_name = "SFace"
        self.input_shape = (112, 112)
        self.output_shape = 128

    def predict(self, img: NDArray[Any]) -> NDArray[Any]:
        """
        Find raw representations of given images. Inputs are fed as opencv's
        FaceRecognizerSF.feature feeds them - in [0, 255] scaled and channels first RGB.
        Args:
            img (np.ndarray): pre-loaded image in BGR with (112, 112, 3) or (n, 112, 112, 3)
                shape and [0, 1] scale
        Returns:
            embeddings (np.ndarray): (n, 128) shaped representations
        """
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        if img.ndim != 4 or img.shape[0] < 1:
            raise ValueError(f"Input image must be (n, 112, 112, 3) shaped but it is {img.shape}")

        # BGR to RGB, channels last to channels first
        input_blob = (img * 255).astype(np.uint8)[..., ::-1].transpose(0, 3, 1, 2)
        input_blob = np.ascontiguousarray(input_blob, dtype=np.float32)

        # the graph is exported with a batch size of 1
        input_name = self.model.get_inputs()[0].name
        embeddings = [
            self.model.run(None, {input_name: input_blob[i : i + 1]})[0]
            for i in range(input_blob.shape[0])
        ]
        return cast(NDArray[Any], np.concatenate(embeddings, axis=0))


def load_model(url: str = WEIGHTS_URL) -> Any:
    """
    Download SFace's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="face_recognition_sface_2021dec.onnx", source_url=url)
