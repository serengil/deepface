# built-in dependencies
from typing import cast

# 3rd party dependencies
import torch
from torch import nn, Tensor

# project dependencies
from deepface.commons import weight_utils
from deepface.models.facial_recognition.pytorch.TorchFacialRecognition import (
    TorchFacialRecognition,
    load_model_weights,
)
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_weights.pth"
)


# pylint: disable=too-few-public-methods
class DeepIdClient(TorchFacialRecognition):
    """
    DeepId model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "DeepId"
        self.input_shape = (47, 55)
        self.output_shape = 160
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)


# pylint: disable=too-many-instance-attributes
class DeepIdNet(nn.Module):  # type: ignore[misc]
    """
    DeepID network finding 160 dimensional representations. Layer names follow the ones
    in the tensorflow backend.

    Notice that fc11 and fc12 expect channels first flattened features, whereas keras
    flattened channels last ones - their weights were reordered accordingly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 20, kernel_size=4)
        self.Conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.Conv3 = nn.Conv2d(40, 60, kernel_size=3)
        self.Conv4 = nn.Conv2d(60, 80, kernel_size=2)

        self.fc11 = nn.Linear(60 * 5 * 4, 160)
        self.fc12 = nn.Linear(80 * 4 * 3, 160)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.99)

    def forward(self, x: Tensor) -> Tensor:
        """
        Find representations of given images
        Args:
            x (Tensor): (n, 3, 55, 47) shaped input in BGR
        Returns:
            embeddings (Tensor): (n, 160) shaped representations
        """
        x = self.dropout(self.pool(self.relu(self.Conv1(x))))
        x = self.dropout(self.pool(self.relu(self.Conv2(x))))
        x = self.dropout(self.pool(self.relu(self.Conv3(x))))

        fc11 = self.fc11(torch.flatten(x, start_dim=1))
        fc12 = self.fc12(torch.flatten(self.relu(self.Conv4(x)), start_dim=1))

        return cast(Tensor, self.relu(fc11 + fc12))


def load_model(
    url: str = WEIGHTS_URL,
) -> DeepIdNet:
    """
    Construct DeepId model, download its weights and load
    Returns:
        model (DeepIdNet)
    """
    model = DeepIdNet()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="deepid_weights.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
