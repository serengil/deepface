# stdlib dependencies
from typing import Any, List, Union

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import cv2
import torch
from torch import nn, Tensor

# project dependencies
from deepface.commons import weight_utils
from deepface.models.demography.pytorch.TorchDemography import TorchDemography
from deepface.models.facial_recognition.pytorch.TorchFacialRecognition import load_model_weights
from deepface.commons.logger import Logger

# Labels for the emotions that can be detected by the model.
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

logger = Logger()

# pylint: disable=line-too-long, disable=too-few-public-methods

WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.pth"

CLASSES = 7


class EmotionClient(TorchDemography):
    """
    Emotion model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "Emotion"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)

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


# pylint: disable=too-many-instance-attributes
class EmotionModel(nn.Module):
    """
    Facial expression model classifying 48x48 grayscale faces into 7 emotions.
    Every convolution and pooling is unpadded, as they are in the tensorflow backend.
    """

    def __init__(self) -> None:
        super().__init__()

        # 1st convolution layer
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=5)
        self.pool_1 = nn.MaxPool2d(kernel_size=5, stride=2)

        # 2nd convolution layer
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool_2 = nn.AvgPool2d(kernel_size=3, stride=2)

        # 3rd convolution layer
        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv_5 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool_3 = nn.AvgPool2d(kernel_size=3, stride=2)

        # fully connected neural networks
        self.fc_1 = nn.Linear(128, 1024)
        self.dropout_1 = nn.Dropout(0.2)
        self.fc_2 = nn.Linear(1024, 1024)
        self.dropout_2 = nn.Dropout(0.2)
        self.fc_3 = nn.Linear(1024, CLASSES)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Find the emotion distribution of given images
        Args:
            x (Tensor): (n, 1, 48, 48) shaped grayscale input
        Returns:
            predictions (Tensor): (n, 7) shaped probabilities, one per emotion
        """
        x = self.pool_1(self.relu(self.conv_1(x)))

        x = self.relu(self.conv_2(x))
        x = self.pool_2(self.relu(self.conv_3(x)))

        x = self.relu(self.conv_4(x))
        x = self.pool_3(self.relu(self.conv_5(x)))

        # (n, 128, 1, 1) to (n, 128)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout_1(self.relu(self.fc_1(x)))
        x = self.dropout_2(self.relu(self.fc_2(x)))

        return torch.softmax(self.fc_3(x), dim=1)


def base_model() -> EmotionModel:
    """
    Base model of the emotion model
    Returns:
        model (EmotionModel)
    """
    return EmotionModel()


def load_model(
    url: str = WEIGHTS_URL,
) -> EmotionModel:
    """
    Consruct emotion model, download and load weights
    Returns:
        model (EmotionModel)
    """
    model = base_model()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="facial_expression_model_weights.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
