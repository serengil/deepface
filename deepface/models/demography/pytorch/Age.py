# stdlib dependencies
from typing import Any, List, Union

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import torch

# project dependencies
from deepface.commons import weight_utils
from deepface.models.demography.pytorch.TorchDemography import (
    TorchDemography,
    VggFaceClassifier,
)
from deepface.models.facial_recognition.pytorch.TorchFacialRecognition import load_model_weights
from deepface.commons.logger import Logger

logger = Logger()

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.pth"
)

# the model predicts a probability for each age in [0, 100]
CLASSES = 101


# pylint: disable=too-few-public-methods
class ApparentAgeClient(TorchDemography):
    """
    Age model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "Age"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)

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
        # import the helper here to avoid circular import issue
        from deepface.models.demography.Age import find_apparent_age

        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Prediction from 3 channels image
        age_predictions = self._predict_internal(imgs)

        # Calculate apparent ages
        if len(age_predictions.shape) == 1:  # Single prediction list
            return find_apparent_age(age_predictions)

        return np.array([find_apparent_age(age_prediction) for age_prediction in age_predictions])


class AgeModel(VggFaceClassifier):
    """
    Age model, predicting a probability for each age in [0, 100]
    """

    def __init__(self) -> None:
        super().__init__(CLASSES)


def base_model() -> AgeModel:
    """
    Base model of the age model
    Returns:
        model (AgeModel)
    """
    return AgeModel()


def load_model(
    url: str = WEIGHTS_URL,
) -> AgeModel:
    """
    Construct age model, download its weights and load
    Returns:
        model (AgeModel)
    """
    model = base_model()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="age_model_weights.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
