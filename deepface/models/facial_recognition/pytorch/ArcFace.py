# built-in dependencies
from typing import Optional, cast

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

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.pth"
)

# batch normalizations are configured as they were in the tensorflow backend, keras'
# momentum is the complement of pytorch's one
BN_EPS = 2e-5
BN_MOMENTUM = 1 - 0.9


# pylint: disable=too-few-public-methods
class ArcFaceClient(TorchFacialRecognition):
    """
    ArcFace model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "ArcFace"
        self.input_shape = (112, 112)
        self.output_shape = 512
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)


def batch_norm(num_features: int) -> nn.BatchNorm2d:
    """
    Batch normalization configured as in the tensorflow backend
    Args:
        num_features (int): number of channels to normalize
    Returns:
        layer (nn.BatchNorm2d)
    """
    return nn.BatchNorm2d(num_features, eps=BN_EPS, momentum=BN_MOMENTUM)


# pylint: disable=too-many-instance-attributes
class Block(nn.Module):  # type: ignore[misc]
    """
    Pre-activation residual unit. Its layers are named after the tensorflow backend,
    e.g. conv_1 stands for the conv2X_blockY_1_conv layer there.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        stride: int = 1,
        conv_shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.conv_0: Optional[nn.Conv2d] = None
        self.bn_0: Optional[nn.BatchNorm2d] = None
        if conv_shortcut:
            self.conv_0 = nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False)
            self.bn_0 = batch_norm(filters)

        self.bn_1 = batch_norm(in_channels)
        self.conv_1 = nn.Conv2d(in_channels, filters, 3, stride=1, padding=1, bias=False)
        self.bn_2 = batch_norm(filters)
        self.prelu_1 = nn.PReLU(filters)
        self.conv_2 = nn.Conv2d(filters, filters, 3, stride=stride, padding=1, bias=False)
        self.bn_3 = batch_norm(filters)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        if self.conv_0 is not None and self.bn_0 is not None:
            shortcut = self.bn_0(self.conv_0(x))

        x = self.bn_1(x)
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.prelu_1(x)
        x = self.conv_2(x)
        x = self.bn_3(x)

        return cast(Tensor, shortcut + x)


def stack(in_channels: int, filters: int, blocks: int, stride: int = 2) -> nn.Sequential:
    """
    A set of residual units, the first one being the only one to downsample
    Args:
        in_channels (int): number of channels the stack is fed with
        filters (int): number of filters of each unit
        blocks (int): number of units
        stride (int): stride of the first unit
    Returns:
        stack (nn.Sequential)
    """
    units = [Block(in_channels, filters, stride=stride)]
    units += [Block(filters, filters, conv_shortcut=False) for _ in range(blocks - 1)]
    return nn.Sequential(*units)


class ResNet34(nn.Module):  # type: ignore[misc]
    """
    ResNet34 backbone of ArcFace
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1_conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.conv1_bn = batch_norm(64)
        self.conv1_prelu = nn.PReLU(64)

        self.conv2 = stack(64, 64, 3)
        self.conv3 = stack(64, 128, 4)
        self.conv4 = stack(128, 256, 6)
        self.conv5 = stack(256, 512, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1_prelu(self.conv1_bn(self.conv1_conv(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return cast(Tensor, self.conv5(x))


class ArcFaceModel(nn.Module):  # type: ignore[misc]
    """
    Final ArcFace model being used for finding embeddings.

    Notice that the bottleneck is fed with channels first flattened feature maps, whereas
    the tensorflow backend flattens them channels last - weights are transferred accordingly.
    """

    def __init__(self) -> None:
        super().__init__()

        self.backbone = ResNet34()
        self.bn = batch_norm(512)
        self.dropout = nn.Dropout(0.4)
        self.dense = nn.Linear(512 * 7 * 7, 512, bias=True)
        self.embedding = nn.BatchNorm1d(512, eps=BN_EPS, momentum=BN_MOMENTUM)

    def forward(self, x: Tensor) -> Tensor:
        """
        Find representations of given images
        Args:
            x (Tensor): (n, 3, 112, 112) shaped input in BGR
        Returns:
            embeddings (Tensor): (n, 512) shaped representations
        """
        x = self.backbone(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return cast(Tensor, self.embedding(x))


def base_model() -> ArcFaceModel:
    """
    Base model of ArcFace
    Returns:
        model (ArcFaceModel)
    """
    return ArcFaceModel()


def load_model(
    url: str = WEIGHTS_URL,
) -> ArcFaceModel:
    """
    Construct ArcFace model, download its weights and load
    Returns:
        model (ArcFaceModel): returning 512 dimensional vectors
    """
    model = base_model()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="arcface_weights.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
