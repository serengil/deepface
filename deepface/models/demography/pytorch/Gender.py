# stdlib dependencies
from typing import Any, List, Union

# 3rd party dependencies
import torch
from numpy.typing import NDArray

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
    "https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.pth"
)

# Labels for the genders that can be detected by the model.
labels = ["Woman", "Man"]

CLASSES = 2


# pylint: disable=too-few-public-methods
class GenderClient(TorchDemography):
    """
    Gender model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "Gender"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)

    def predict(self, img: Union[NDArray[Any], List[NDArray[Any]]]) -> NDArray[Any]:
        """
        Predict gender probabilities for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (n, 2)
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Prediction
        predictions = self._predict_internal(imgs)

        return predictions


class GenderModel(VggFaceClassifier):
    """
    Gender model, predicting a probability for each of the labels above
    """

    def __init__(self) -> None:
        super().__init__(CLASSES)


def base_model() -> GenderModel:
    """
    Base model of the gender model
    Returns:
        model (GenderModel)
    """
    return GenderModel()


def load_model(
    url: str = WEIGHTS_URL,
) -> GenderModel:
    """
    Construct gender model, download its weights and load
    Returns:
        model (GenderModel)
    """
    model = base_model()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="gender_model_weights.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
