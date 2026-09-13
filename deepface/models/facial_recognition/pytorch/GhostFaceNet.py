# built-in dependencies
from typing import Optional, cast

# 3rd party dependencies
import torch
from torch import nn, Tensor
from torch.nn import functional as F

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
    "https://github.com/serengil/deepface_models/releases/download/v1.0/ghostfacenet_v1.pth"
)

# batch normalizations keep keras' defaults, whose momentum is pytorch's complement
BN_EPS = 0.001
BN_MOMENTUM = 1 - 0.99

# GhostFaceNetV1 architecture, as it is in the tensorflow backend
DW_KERNELS = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
EXPS = [20, 64, 92, 92, 156, 312, 260, 240, 240, 624, 872, 872, 1248, 1248, 1248, 664]
OUTS = [20, 32, 32, 52, 52, 104, 104, 104, 104, 144, 144, 208, 208, 208, 208, 208]
STRIDES = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
REDUCTIONS = [0, 0, 0, 24, 40, 0, 0, 0, 0, 156, 220, 220, 0, 312, 0, 168]

STEM_CHANNELS = 20
TAIL_CHANNELS = 664
EMBEDDING_SIZE = 512


# pylint: disable=too-few-public-methods
class GhostFaceNetClient(TorchFacialRecognition):
    """
    GhostFaceNet model (GhostFaceNetV1 backbone) - pytorch backend
    Repo: https://github.com/HamadYA/GhostFaceNets
    Pre-trained weights: https://github.com/HamadYA/GhostFaceNets/releases/tag/v1.2
        Author declared that this backbone and pre-trained weights got 99.7667% accuracy on LFW
    """

    def __init__(self) -> None:
        self.model_name = "GhostFaceNet"
        self.input_shape = (112, 112)
        self.output_shape = 512
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)


def hard_sigmoid(x: Tensor) -> Tensor:
    """
    Keras' hard sigmoid - notice that it is steeper than pytorch's nn.Hardsigmoid,
    which divides by 6 instead of multiplying by 0.2
    Args:
        x (Tensor): input tensor
    Returns:
        activated (Tensor): 0.2 * x + 0.5 clipped into [0, 1]
    """
    return torch.clamp(torch.add(torch.mul(x, 0.2), 0.5), min=0.0, max=1.0)


def same_padding(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    """
    Pad an input as tensorflow's "same" padding does. Notice that tensorflow pads the
    bottom and the right hand side more than the top and the left hand side when an odd
    number of pixels is missing, whereas pytorch's padding argument is always symmetric.
    Args:
        x (Tensor): (n, c, h, w) shaped input
        kernel_size (int): kernel size of the convolution to be applied
        stride (int): stride of the convolution to be applied
    Returns:
        padded (Tensor): zero padded input
    """
    height, width = x.shape[-2:]
    pad_height = max((-(-height // stride) - 1) * stride + kernel_size - height, 0)
    pad_width = max((-(-width // stride) - 1) * stride + kernel_size - width, 0)
    return F.pad(
        x,
        [pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2],
    )


class ConvBn(nn.Module):  # type: ignore[misc]
    """
    Convolution without bias, followed by a batch normalization and an optional prelu.
    Setting depthwise applies the convolution per channel as keras' DepthwiseConv2D does.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        depthwise: bool = False,
        activation: bool = True,
        pad_same: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_same = pad_same

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels if depthwise else 1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.prelu = nn.PReLU(out_channels) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        if self.pad_same:
            x = same_padding(x, self.kernel_size, self.stride)
        x = self.bn(self.conv(x))
        if self.prelu is not None:
            x = self.prelu(x)
        return x


class SqueezeExcite(nn.Module):  # type: ignore[misc]
    """
    Squeeze and excitation block gating the channels with a hard sigmoid
    """

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(channels, reduction, kernel_size=1, bias=True)
        self.prelu = nn.PReLU(reduction)
        self.expand = nn.Conv2d(reduction, channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        se = F.adaptive_avg_pool2d(x, 1)
        se = self.prelu(self.reduce(se))
        se = hard_sigmoid(self.expand(se))
        return cast(Tensor, x * se)


class GhostModule(nn.Module):  # type: ignore[misc]
    """
    Ghost module generating half of its feature maps with a pointwise convolution and
    the other half cheaply, with a depthwise convolution over the first half.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: bool = True) -> None:
        super().__init__()
        primary_channels = out_channels // 2
        self.primary = ConvBn(in_channels, primary_channels, 1, activation=activation)
        self.cheap = ConvBn(
            primary_channels, primary_channels, 3, depthwise=True, activation=activation
        )

    def forward(self, x: Tensor) -> Tensor:
        primary = self.primary(x)
        return torch.cat([primary, self.cheap(primary)], dim=1)


class GhostBottleneck(nn.Module):  # type: ignore[misc]
    """
    Ghost bottleneck - expanding ghost module, optional downsampling depthwise convolution,
    optional squeeze and excitation, projecting ghost module and an optional shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        dw_kernel: int,
        stride: int,
        exp: int,
        out_channels: int,
        reduction: int,
        shortcut: bool,
    ) -> None:
        super().__init__()

        self.ghost_1 = GhostModule(in_channels, exp, activation=True)
        self.downsample = (
            ConvBn(exp, exp, dw_kernel, stride=stride, depthwise=True, activation=False)
            if stride > 1
            else None
        )
        self.se = SqueezeExcite(exp, reduction) if reduction > 0 else None
        self.ghost_2 = GhostModule(exp, out_channels, activation=False)

        self.shortcut: Optional[nn.Sequential] = None
        if shortcut:
            self.shortcut = nn.Sequential(
                ConvBn(
                    in_channels,
                    in_channels,
                    dw_kernel,
                    stride=stride,
                    depthwise=True,
                    activation=False,
                ),
                ConvBn(in_channels, out_channels, 1, activation=False, pad_same=False),
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)

        x = self.ghost_1(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost_2(x)

        return cast(Tensor, residual + x)


class GhostFaceNetV1(nn.Module):  # type: ignore[misc]
    """
    GhostFaceNetV1 model. Refactored from
        github.com/HamadYA/GhostFaceNets/blob/main/backbones/ghost_model.py
    """

    def __init__(self) -> None:
        super().__init__()

        self.stem = ConvBn(3, STEM_CHANNELS, 3, activation=True)

        bottlenecks = []
        in_channels = STEM_CHANNELS
        for dw_kernel, stride, exp, out_channels, reduction in zip(
            DW_KERNELS, STRIDES, EXPS, OUTS, REDUCTIONS
        ):
            bottlenecks.append(
                GhostBottleneck(
                    in_channels,
                    dw_kernel,
                    stride,
                    exp,
                    out_channels,
                    reduction,
                    shortcut=not (out_channels == in_channels and stride == 1),
                )
            )
            in_channels = out_channels
        self.bottlenecks = nn.Sequential(*bottlenecks)

        self.tail = ConvBn(in_channels, TAIL_CHANNELS, 1, activation=True, pad_same=False)

        # global depthwise convolution reducing the 7x7 feature maps into 1x1 ones
        self.gdc_dw = nn.Conv2d(
            TAIL_CHANNELS, TAIL_CHANNELS, kernel_size=7, groups=TAIL_CHANNELS, bias=False
        )
        self.gdc_bn = nn.BatchNorm2d(TAIL_CHANNELS, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.gdc_conv = nn.Conv2d(TAIL_CHANNELS, EMBEDDING_SIZE, kernel_size=1, bias=True)
        self.pre_embedding = nn.BatchNorm1d(EMBEDDING_SIZE, eps=BN_EPS, momentum=BN_MOMENTUM)

    def forward(self, x: Tensor) -> Tensor:
        """
        Find representations of given images
        Args:
            x (Tensor): (n, 3, 112, 112) shaped input in BGR
        Returns:
            embeddings (Tensor): (n, 512) shaped representations
        """
        x = self.stem(x)
        x = self.bottlenecks(x)
        x = self.tail(x)

        x = self.gdc_bn(self.gdc_dw(x))
        x = self.gdc_conv(x)
        x = torch.flatten(x, start_dim=1)
        return cast(Tensor, self.pre_embedding(x))


def base_model() -> GhostFaceNetV1:
    """
    Base model of GhostFaceNet
    Returns:
        model (GhostFaceNetV1)
    """
    return GhostFaceNetV1()


def load_model(
    url: str = WEIGHTS_URL,
) -> GhostFaceNetV1:
    """
    Construct GhostFaceNet model, download its weights and load
    Returns:
        model (GhostFaceNetV1): returning 512 dimensional vectors
    """
    model = base_model()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="ghostfacenet_v1.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
