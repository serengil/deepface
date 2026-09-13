# built-in dependencies
from typing import Tuple, Union, cast

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

# pylint:disable=line-too-long
FACENET128_WEIGHTS = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.pth"
)
FACENET512_WEIGHTS = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.pth"
)

# batch normalizations are configured as they were in the tensorflow backend. keras' momentum
# is the complement of pytorch's one, and scale=False means gamma is not learned but kept as 1.
BN_EPS = 0.001
BN_MOMENTUM = 1 - 0.995

# --------------------------------


# pylint: disable=too-few-public-methods
class FaceNet128dClient(TorchFacialRecognition):
    """
    FaceNet-128d model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "FaceNet-128d"
        self.input_shape = (160, 160)
        self.output_shape = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_facenet128d_model().to(self.device)


class FaceNet512dClient(TorchFacialRecognition):
    """
    FaceNet-512d model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "FaceNet-512d"
        self.input_shape = (160, 160)
        self.output_shape = 512
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_facenet512d_model().to(self.device)


class ConvBn(nn.Module):  # type: ignore[misc]
    """
    Convolution without bias, followed by a batch normalization and a relu activation.
    Gamma of the batch normalization is not learned, it is kept as 1 - as scale=False
    was set in the tensorflow backend.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.bn.weight.data.fill_(1)
        self.bn.weight.requires_grad_(False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.relu(self.bn(self.conv(x))))


class Block35(nn.Module):  # type: ignore[misc]
    """
    35x35 grid sized inception-resnet-A block
    """

    def __init__(self, scale: float = 0.17) -> None:
        super().__init__()
        self.scale = scale

        self.branch_0 = ConvBn(256, 32, 1)
        self.branch_1 = nn.Sequential(
            ConvBn(256, 32, 1),
            ConvBn(32, 32, 3, padding=1),
        )
        self.branch_2 = nn.Sequential(
            ConvBn(256, 32, 1),
            ConvBn(32, 32, 3, padding=1),
            ConvBn(32, 32, 3, padding=1),
        )
        self.up = nn.Conv2d(96, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        mixed = torch.cat([self.branch_0(x), self.branch_1(x), self.branch_2(x)], dim=1)
        return cast(Tensor, self.relu(x + self.scale * self.up(mixed)))


class Block17(nn.Module):  # type: ignore[misc]
    """
    17x17 grid sized inception-resnet-B block
    """

    def __init__(self, scale: float = 0.1) -> None:
        super().__init__()
        self.scale = scale

        self.branch_0 = ConvBn(896, 128, 1)
        self.branch_1 = nn.Sequential(
            ConvBn(896, 128, 1),
            ConvBn(128, 128, (1, 7), padding=(0, 3)),
            ConvBn(128, 128, (7, 1), padding=(3, 0)),
        )
        self.up = nn.Conv2d(256, 896, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        mixed = torch.cat([self.branch_0(x), self.branch_1(x)], dim=1)
        return cast(Tensor, self.relu(x + self.scale * self.up(mixed)))


class Block8(nn.Module):  # type: ignore[misc]
    """
    8x8 grid sized inception-resnet-C block. The last one of the network is not activated
    and does not scale its residual down.
    """

    def __init__(self, scale: float = 0.2, activation: bool = True) -> None:
        super().__init__()
        self.scale = scale

        self.branch_0 = ConvBn(1792, 192, 1)
        self.branch_1 = nn.Sequential(
            ConvBn(1792, 192, 1),
            ConvBn(192, 192, (1, 3), padding=(0, 1)),
            ConvBn(192, 192, (3, 1), padding=(1, 0)),
        )
        self.up = nn.Conv2d(384, 1792, kernel_size=1)
        self.relu = nn.ReLU(inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        mixed = torch.cat([self.branch_0(x), self.branch_1(x)], dim=1)
        x = x + self.scale * self.up(mixed)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Mixed6a(nn.Module):  # type: ignore[misc]
    """
    Reduction-A block shrinking 35x35 grids into 17x17 ones
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = ConvBn(256, 384, 3, stride=2)
        self.branch_1 = nn.Sequential(
            ConvBn(256, 192, 1),
            ConvBn(192, 192, 3, padding=1),
            ConvBn(192, 256, 3, stride=2),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.branch_0(x), self.branch_1(x), self.branch_2(x)], dim=1)


class Mixed7a(nn.Module):  # type: ignore[misc]
    """
    Reduction-B block shrinking 17x17 grids into 8x8 ones
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = nn.Sequential(
            ConvBn(896, 256, 1),
            ConvBn(256, 384, 3, stride=2),
        )
        self.branch_1 = nn.Sequential(
            ConvBn(896, 256, 1),
            ConvBn(256, 256, 3, stride=2),
        )
        self.branch_2 = nn.Sequential(
            ConvBn(896, 256, 1),
            ConvBn(256, 256, 3, padding=1),
            ConvBn(256, 256, 3, stride=2),
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [self.branch_0(x), self.branch_1(x), self.branch_2(x), self.branch_3(x)], dim=1
        )


# pylint: disable=too-many-instance-attributes
class InceptionResNetV1(nn.Module):  # type: ignore[misc]
    """
    InceptionResNetV1 model heavily inspired from
    github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py
    As mentioned in Sandberg's repo's readme, pre-trained models are using Inception ResNet v1
    Besides training process is documented at
    sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/

    Args:
        dimension (int): number of dimensions in the embedding layer
    """

    def __init__(self, dimension: int = 128) -> None:
        super().__init__()

        # stem
        self.conv2d_1a_3x3 = ConvBn(3, 32, 3, stride=2)
        self.conv2d_2a_3x3 = ConvBn(32, 32, 3)
        self.conv2d_2b_3x3 = ConvBn(32, 64, 3, padding=1)
        self.maxpool_3a_3x3 = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b_1x1 = ConvBn(64, 80, 1)
        self.conv2d_4a_3x3 = ConvBn(80, 192, 3)
        self.conv2d_4b_3x3 = ConvBn(192, 256, 3, stride=2)

        self.block35 = nn.ModuleList([Block35(scale=0.17) for _ in range(5)])
        self.mixed_6a = Mixed6a()
        self.block17 = nn.ModuleList([Block17(scale=0.1) for _ in range(10)])
        self.mixed_7a = Mixed7a()
        # the 6th block8 neither scales its residual down nor is activated
        self.block8 = nn.ModuleList(
            [Block8(scale=0.2) for _ in range(5)] + [Block8(scale=1.0, activation=False)]
        )

        # classification block
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1.0 - 0.8)
        self.bottleneck = nn.Linear(1792, dimension, bias=False)
        self.bottleneck_bn = nn.BatchNorm1d(dimension, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.bottleneck_bn.weight.data.fill_(1)
        self.bottleneck_bn.weight.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Find representations of given images
        Args:
            x (Tensor): (n, 3, 160, 160) shaped input in BGR
        Returns:
            embeddings (Tensor): (n, dimension) shaped representations
        """
        x = self.conv2d_1a_3x3(x)
        x = self.conv2d_2a_3x3(x)
        x = self.conv2d_2b_3x3(x)
        x = self.maxpool_3a_3x3(x)
        x = self.conv2d_3b_1x1(x)
        x = self.conv2d_4a_3x3(x)
        x = self.conv2d_4b_3x3(x)

        for block in self.block35:
            x = block(x)
        x = self.mixed_6a(x)
        for block in self.block17:
            x = block(x)
        x = self.mixed_7a(x)
        for block in self.block8:
            x = block(x)

        x = torch.flatten(self.avgpool(x), start_dim=1)
        x = self.dropout(x)
        x = self.bottleneck(x)
        return cast(Tensor, self.bottleneck_bn(x))


def load_facenet128d_model(
    url: str = FACENET128_WEIGHTS,
) -> InceptionResNetV1:
    """
    Construct FaceNet-128d model, download weights and then load weights
    Returns:
        model (InceptionResNetV1)
    """
    model = InceptionResNetV1()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="facenet_weights.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)


def load_facenet512d_model(
    url: str = FACENET512_WEIGHTS,
) -> InceptionResNetV1:
    """
    Construct FaceNet-512d model, download its weights and load
    Returns:
        model (InceptionResNetV1)
    """
    model = InceptionResNetV1(dimension=512)

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="facenet512_weights.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
