# stdlib dependencies
from typing import Any, List, Union

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.models.demography.DemographyUtils import find_apparent_age
from deepface.models.demography.onnx.OnnxDemography import OnnxDemography
from deepface.commons.onnx_utils import build_session
from deepface.commons.logger import Logger

logger = Logger()

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/age_model_weights.onnx",
]


# pylint: disable=too-few-public-methods
class ApparentAgeClient(OnnxDemography):
    """
    Age model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "Age"
        self.model = load_model()

    def predict(
        self, img: Union[NDArray[Any], List[NDArray[Any]]]
    ) -> Union[np.float64, NDArray[Any]]:
        """
        Predict apparent age(s) for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (age_classes,) if single image,
            np.ndarray (n, age_classes) if batched images.
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Prediction from 3 channels image
        age_predictions = self._predict_internal(imgs)

        # Calculate apparent ages
        if len(age_predictions.shape) == 1:  # Single prediction list
            return find_apparent_age(age_predictions)

        return np.array([find_apparent_age(age_prediction) for age_prediction in age_predictions])


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download age model's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="age_model_weights.onnx", source_url=url)
