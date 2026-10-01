# stdlib dependencies
from typing import Any, List, Union

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import cv2

# project dependencies
from deepface.models.demography.DemographyUtils import EMOTION_LABELS
from deepface.models.demography.onnx.OnnxDemography import OnnxDemography
from deepface.commons.onnx_utils import build_session
from deepface.commons.logger import Logger

# Labels for the emotions that can be detected by the model.
labels = EMOTION_LABELS

logger = Logger()

# pylint: disable=line-too-long, disable=too-few-public-methods

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/facial_expression_model_weights.onnx",
]


class EmotionClient(OnnxDemography):
    """
    Emotion model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "Emotion"
        self.model = load_model()

    def _preprocess_image(self, img: NDArray[Any]) -> NDArray[Any]:
        """
        Preprocess single image for emotion detection
        Args:
            img: Input image (224, 224, 3)
        Returns:
            Preprocessed grayscale image (48, 48)
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        return img_gray

    def predict(self, img: Union[NDArray[Any], List[NDArray[Any]]]) -> NDArray[Any]:
        """
        Predict emotion probabilities for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (n, n_emotions)
            where n_emotions is the number of emotion categories
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        processed_imgs = np.expand_dims(
            np.array([self._preprocess_image(img) for img in imgs]), axis=-1
        )

        # Prediction
        predictions = self._predict_internal(processed_imgs)

        return predictions


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download emotion model's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="facial_expression_model_weights.onnx", source_url=url)
