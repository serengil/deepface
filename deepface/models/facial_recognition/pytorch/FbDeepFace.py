# built-in dependencies
from typing import cast

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

# pylint: disable=line-too-long, too-few-public-methods
WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/VGGFace2_DeepFace_weights_val-0.9034.pth"


class DeepFaceClient(TorchFacialRecognition):
    """
    Fb's DeepFace model class - pytorch backend.

    Unlike the tensorflow backend, it does not require an old framework version because
    locally connected layers are implemented here.
    """

    def __init__(self) -> None:
        self.model_name = "DeepFace"
        self.input_shape = (152, 152)
        self.output_shape = 4096
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)


class LocallyConnected2d(nn.Module):  # type: ignore[misc]
    """
    Convolution-like layer whose filters are not shared - every output location has its own
    weights, as keras' LocallyConnected2D does. Padding is not supported, as it was not used.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of filters at each output location
        input_size (int): height and width of the square input
        kernel_size (int): height and width of the square kernel
        stride (int): stride of the sliding window
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = (input_size - kernel_size) // stride + 1

        locations = self.output_size * self.output_size
        patch_size = in_channels * kernel_size * kernel_size

        # (locations, out_channels, patch_size) with channels first ordered patches
        self.weight = nn.Parameter(torch.empty(locations, out_channels, patch_size))
        self.bias = nn.Parameter(torch.zeros(out_channels, self.output_size, self.output_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        # (n, patch_size, locations)
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        # (n, out_channels, locations)
        out = torch.einsum("npl,lfp->nfl", patches, self.weight)
        out = out.reshape(x.shape[0], -1, self.output_size, self.output_size)
        return cast(Tensor, out + self.bias)


# pylint: disable=too-many-instance-attributes
class DeepFaceNet(nn.Module):  # type: ignore[misc]
    """
    Fb's DeepFace network finding 4096 dimensional representations. Layer names follow the
    ones in the tensorflow backend. F8, the 8631 way classifier on top of the representation
    layer F7, is not built since it is not used to find embeddings.

    Notice that F7 expects channels first flattened features, whereas keras flattened
    channels last ones - its weights were reordered accordingly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.C1 = nn.Conv2d(3, 32, kernel_size=11)
        self.C3 = nn.Conv2d(32, 16, kernel_size=9)
        self.L4 = LocallyConnected2d(16, 16, input_size=63, kernel_size=9)
        self.L5 = LocallyConnected2d(16, 16, input_size=55, kernel_size=7, stride=2)
        self.L6 = LocallyConnected2d(16, 16, input_size=25, kernel_size=5)
        self.F7 = nn.Linear(16 * 21 * 21, 4096)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Find representations of given images
        Args:
            x (Tensor): (n, 3, 152, 152) shaped input in BGR
        Returns:
            embeddings (Tensor): (n, 4096) shaped representations
        """
        x = self.relu(self.C1(x))
        # max pooling with same padding - tf pads 142x142 inputs by 1 at the bottom and right
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=float("-inf")), kernel_size=3, stride=2)
        x = self.relu(self.C3(x))
        x = self.relu(self.L4(x))
        x = self.relu(self.L5(x))
        x = self.relu(self.L6(x))
        x = torch.flatten(x, start_dim=1)
        return cast(Tensor, self.relu(self.F7(x)))


def load_model(
    url: str = WEIGHTS_URL,
) -> DeepFaceNet:
    """
    Construct DeepFace model, download its weights and load
    Returns:
        model (DeepFaceNet)
    """
    model = DeepFaceNet()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="VGGFace2_DeepFace_weights_val-0.9034.pth", source_url=url
    )

    return load_model_weights(model=model, weight_file=weight_file)
