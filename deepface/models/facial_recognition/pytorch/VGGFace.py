# built-in dependencies
from typing import Any, List, cast

# 3rd party dependencies
from numpy.typing import NDArray
import torch
from torch import nn, Tensor

# project dependencies
from deepface.commons import weight_utils
from deepface.modules import verification
from deepface.models.facial_recognition.pytorch.TorchFacialRecognition import (
    TorchFacialRecognition,
    load_model_weights,
)
from deepface.commons.logger import Logger

logger = Logger()

# ---------------------------------------

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.pth"
)


# pylint: disable=too-few-public-methods
class VggFaceClient(TorchFacialRecognition):
    """
    VGG-Face model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "VGG-Face"
        self.input_shape = (224, 224)
        self.output_shape = 4096
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)

    def forward(self, img: NDArray[Any]) -> List[float]:
        """
        Generates embeddings using the VGG-Face model.
            This method incorporates an additional normalization layer.

        Args:
            img (np.ndarray): pre-loaded image in BGR with (224, 224, 3)
                or (n, 224, 224, 3) shape - channels last as in the tensorflow backend
        Returns
            embeddings (list): multi-dimensional vector
        """
        embeddings = self.predict(img)

        if embeddings.shape[0] == 1:
            embedding_norm = verification.l2_normalize(embeddings[0].tolist())
        else:
            embedding_norm = verification.l2_normalize(embeddings.tolist(), axis=1)

        return cast(List[float], embedding_norm.tolist())


# pylint: disable=too-many-instance-attributes
class VggFaceBaseModel(nn.Module):
    """
    Base model of VGG-Face being used for classification - not to find embeddings.
    It was trained to classify 2622 identities.

    Layer names follow the original VGG-Face (matconvnet) naming, fc6 / fc7 / fc8
    being implemented as convolutions as in the tensorflow backend.

    Args:
        include_top (bool): build the 2622 way classifier on top. Models reusing the
            trunk only - the age, gender and race ones - are built without it.
    """

    def __init__(self, include_top: bool = True) -> None:
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.fc8 = nn.Conv2d(4096, 2622, kernel_size=1) if include_top else None

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def embed(self, x: Tensor) -> Tensor:
        """
        Find 4096 dimensional representations of given images
        Args:
            x (Tensor): (n, 3, 224, 224) shaped input in BGR
        Returns:
            embeddings (Tensor): (n, 4096) shaped representations
        """
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))

        # (n, 4096, 1, 1) to (n, 4096)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Classify given images into 2622 identities
        Args:
            x (Tensor): (n, 3, 224, 224) shaped input in BGR
        Returns:
            predictions (Tensor): (n, 2622) shaped class probabilities
        """
        if self.fc8 is None:
            raise ValueError("this model was built without its classifier, use embed instead")

        x = self.embed(x)
        x = self.dropout(x)
        x = self.fc8(x[:, :, None, None])
        return torch.softmax(torch.flatten(x, start_dim=1), dim=1)


class VggFaceDescriptor(nn.Module):
    """
    Final VGG-Face model being used for finding embeddings.

    The 4096 dimensional model offers 6% to 14% increasement on accuracy over the
    2622 dimensional one, softmax causes underfitting.
    See https://github.com/serengil/deepface/issues/944
    """

    def __init__(self, model: VggFaceBaseModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.embed(x)


def base_model() -> VggFaceBaseModel:
    """
    Base model of VGG-Face being used for classification - not to find embeddings
    Returns:
        model (VggFaceBaseModel): model was trained to classify 2622 identities
    """
    return VggFaceBaseModel()


def load_model(
    url: str = WEIGHTS_URL,
) -> VggFaceDescriptor:
    """
    Final VGG-Face model being used for finding embeddings
    Returns:
        model (VggFaceDescriptor): returning 4096 dimensional vectors
    """
    model = base_model()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="vgg_face_weights.pth", source_url=url
    )

    model = load_model_weights(model=model, weight_file=weight_file)

    descriptor = VggFaceDescriptor(model)
    descriptor.eval()

    return descriptor
