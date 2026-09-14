# built-in dependencies
from typing import Tuple, Union, cast

# 3rd party dependencies
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# project dependencies
from deepface.commons import weight_utils
from deepface.models.facial_recognition.pytorch.TorchFacialRecognition import (
    TorchFacialRecognition,
    load_model_weights,
)
from deepface.commons.logger import Logger

logger = Logger()

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.pth"
)

# batch normalizations are configured as they were in the tensorflow backend. keras' momentum
# is the complement of pytorch's one.
BN_EPS = 0.00001
BN_MOMENTUM = 1 - 0.99

# ---------------------------------------


# pylint: disable=too-few-public-methods
class OpenFaceClient(TorchFacialRecognition):
    """
    OpenFace model class - pytorch backend
    """

    def __init__(self) -> None:
        self.model_name = "OpenFace"
        self.input_shape = (96, 96)
        self.output_shape = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)


class ConvBn(nn.Module):  # type: ignore[misc]
    """
    Convolution followed by a batch normalization and a relu activation. Zero padding is
    applied before the convolution as ZeroPadding2D layers did in the tensorflow backend.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.relu(self.bn(self.conv(x))))


def lrn(x: Tensor) -> Tensor:
    """
    Local response normalization with tf.nn.lrn's defaults - depth_radius=5 and bias=1.
    tf sums squares over 2 * depth_radius + 1 channels whereas pytorch averages them,
    so alpha is scaled by the window size.
    """
    size = 2 * 5 + 1
    return F.local_response_norm(x, size=size, alpha=1e-4 * size, beta=0.75, k=1.0)


def l2_pool(x: Tensor) -> Tensor:
    """
    L2 pooling of 3x3 windows with stride 3 - sqrt of the sum of squares in a window
    """
    return torch.sqrt(F.avg_pool2d(x * x, kernel_size=3, stride=3).mul(9))


class Inception3a(nn.Module):  # type: ignore[misc]
    """
    Inception block 3a - (n, 192, 12, 12) to (n, 256, 12, 12)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_3x3 = nn.Sequential(ConvBn(192, 96, 1), ConvBn(96, 128, 3, padding=1))
        self.branch_5x5 = nn.Sequential(ConvBn(192, 16, 1), ConvBn(16, 32, 5, padding=2))
        self.branch_pool = ConvBn(192, 32, 1)
        self.branch_1x1 = ConvBn(192, 64, 1)

    def forward(self, x: Tensor) -> Tensor:
        pool = self.branch_pool(F.max_pool2d(x, kernel_size=3, stride=2))
        pool = F.pad(pool, (3, 4, 3, 4))
        return torch.cat([self.branch_3x3(x), self.branch_5x5(x), pool, self.branch_1x1(x)], dim=1)


class Inception3b(nn.Module):  # type: ignore[misc]
    """
    Inception block 3b - (n, 256, 12, 12) to (n, 320, 12, 12)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_3x3 = nn.Sequential(ConvBn(256, 96, 1), ConvBn(96, 128, 3, padding=1))
        self.branch_5x5 = nn.Sequential(ConvBn(256, 32, 1), ConvBn(32, 64, 5, padding=2))
        self.branch_pool = ConvBn(256, 64, 1)
        self.branch_1x1 = ConvBn(256, 64, 1)

    def forward(self, x: Tensor) -> Tensor:
        pool = F.pad(self.branch_pool(l2_pool(x)), (4, 4, 4, 4))
        return torch.cat([self.branch_3x3(x), self.branch_5x5(x), pool, self.branch_1x1(x)], dim=1)


class Inception3c(nn.Module):  # type: ignore[misc]
    """
    Inception block 3c - (n, 320, 12, 12) to (n, 640, 6, 6)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_3x3 = nn.Sequential(
            ConvBn(320, 128, 1), ConvBn(128, 256, 3, stride=2, padding=1)
        )
        self.branch_5x5 = nn.Sequential(ConvBn(320, 32, 1), ConvBn(32, 64, 5, stride=2, padding=2))

    def forward(self, x: Tensor) -> Tensor:
        pool = F.pad(F.max_pool2d(x, kernel_size=3, stride=2), (0, 1, 0, 1))
        return torch.cat([self.branch_3x3(x), self.branch_5x5(x), pool], dim=1)


class Inception4a(nn.Module):  # type: ignore[misc]
    """
    Inception block 4a - (n, 640, 6, 6) to (n, 640, 6, 6)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_3x3 = nn.Sequential(ConvBn(640, 96, 1), ConvBn(96, 192, 3, padding=1))
        self.branch_5x5 = nn.Sequential(ConvBn(640, 32, 1), ConvBn(32, 64, 5, padding=2))
        self.branch_pool = ConvBn(640, 128, 1)
        self.branch_1x1 = ConvBn(640, 256, 1)

    def forward(self, x: Tensor) -> Tensor:
        pool = F.pad(self.branch_pool(l2_pool(x)), (2, 2, 2, 2))
        return torch.cat([self.branch_3x3(x), self.branch_5x5(x), pool, self.branch_1x1(x)], dim=1)


class Inception4e(nn.Module):  # type: ignore[misc]
    """
    Inception block 4e - (n, 640, 6, 6) to (n, 1024, 3, 3)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_3x3 = nn.Sequential(
            ConvBn(640, 160, 1), ConvBn(160, 256, 3, stride=2, padding=1)
        )
        self.branch_5x5 = nn.Sequential(ConvBn(640, 64, 1), ConvBn(64, 128, 5, stride=2, padding=2))

    def forward(self, x: Tensor) -> Tensor:
        pool = F.pad(F.max_pool2d(x, kernel_size=3, stride=2), (0, 1, 0, 1))
        return torch.cat([self.branch_3x3(x), self.branch_5x5(x), pool], dim=1)


class Inception5a(nn.Module):  # type: ignore[misc]
    """
    Inception block 5a - (n, 1024, 3, 3) to (n, 736, 3, 3)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_3x3 = nn.Sequential(ConvBn(1024, 96, 1), ConvBn(96, 384, 3, padding=1))
        self.branch_pool = ConvBn(1024, 96, 1)
        self.branch_1x1 = ConvBn(1024, 256, 1)

    def forward(self, x: Tensor) -> Tensor:
        pool = F.pad(self.branch_pool(l2_pool(x)), (1, 1, 1, 1))
        return torch.cat([self.branch_3x3(x), pool, self.branch_1x1(x)], dim=1)


class Inception5b(nn.Module):  # type: ignore[misc]
    """
    Inception block 5b - (n, 736, 3, 3) to (n, 736, 3, 3)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_3x3 = nn.Sequential(ConvBn(736, 96, 1), ConvBn(96, 384, 3, padding=1))
        self.branch_pool = ConvBn(736, 96, 1)
        self.branch_1x1 = ConvBn(736, 256, 1)

    def forward(self, x: Tensor) -> Tensor:
        pool = self.branch_pool(F.max_pool2d(x, kernel_size=3, stride=2))
        pool = F.pad(pool, (1, 1, 1, 1))
        return torch.cat([self.branch_3x3(x), pool, self.branch_1x1(x)], dim=1)


# pylint: disable=too-many-instance-attributes
class OpenFaceNet(nn.Module):  # type: ignore[misc]
    """
    OpenFace nn4.small2 network finding 128 dimensional l2 normalized representations
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBn(3, 64, 7, stride=2, padding=3)
        self.conv2 = ConvBn(64, 64, 1)
        self.conv3 = ConvBn(64, 192, 3, padding=1)

        self.inception_3a = Inception3a()
        self.inception_3b = Inception3b()
        self.inception_3c = Inception3c()
        self.inception_4a = Inception4a()
        self.inception_4e = Inception4e()
        self.inception_5a = Inception5a()
        self.inception_5b = Inception5b()

        self.dense_layer = nn.Linear(736, 128)

    def forward(self, x: Tensor) -> Tensor:
        """
        Find representations of given images
        Args:
            x (Tensor): (n, 3, 96, 96) shaped input in BGR
        Returns:
            embeddings (Tensor): (n, 128) shaped l2 normalized representations
        """
        x = self.conv1(x)
        x = F.max_pool2d(F.pad(x, (1, 1, 1, 1)), kernel_size=3, stride=2)
        x = lrn(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = lrn(x)
        x = F.max_pool2d(F.pad(x, (1, 1, 1, 1)), kernel_size=3, stride=2)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.inception_3c(x)
        x = self.inception_4a(x)
        x = self.inception_4e(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = torch.flatten(F.avg_pool2d(x, kernel_size=3, stride=1), start_dim=1)
        x = self.dense_layer(x)

        # as keras' l2_normalize does, squared sum is clipped by 1e-12 instead of the norm
        squared_sum = torch.sum(x * x, dim=1, keepdim=True).clamp(min=1e-12)
        return cast(Tensor, x * torch.rsqrt(squared_sum))


def load_model(
    url: str = WEIGHTS_URL,
) -> OpenFaceNet:
    """
    Construct OpenFace model, download its weights and load
    Returns:
        model (OpenFaceNet)
    """
    model = OpenFaceNet()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="openface_weights.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
