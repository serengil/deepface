# built-in dependencies
from typing import Any, cast

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn, Tensor

# project dependencies
from deepface.models.Demography import Demography
from deepface.models.facial_recognition.pytorch.VGGFace import VggFaceBaseModel
from deepface.commons.logger import Logger

logger = Logger()

# Notice that all facial attribute analysis models with pytorch backend must be
# inherited from this class


# pylint: disable=too-few-public-methods
class TorchDemography(Demography):
    """
    Base class of facial attribute analysis models having pytorch backend.

    Inputs are still expected to be channels last - (n, X, X, c) shaped - as in the
    tensorflow backend, they are transposed to channels first internally.
    """

    model: nn.Module
    device: torch.device

    def _predict_internal(self, img_batch: NDArray[Any]) -> NDArray[Any]:
        """
        Predict for single image or batched images.
        Args:
            img_batch:
                Batch of images as np.ndarray (n, x, y, c)
                    with n >= 1, x = image width, y = image height, c = channel
                Or Single image as np.ndarray (1, x, y, c)
                    with x = image width, y = image height and c = channel
                The channel dimension will be 1 if input is grayscale. (For emotion model)
        Returns:
            predictions (np.ndarray): (classes,) for a single image, (n, classes) otherwise
        """
        if not self.model_name:  # Check if called from derived class
            raise NotImplementedError("no model selected")
        assert img_batch.ndim == 4, "expected 4-dimensional tensor input"

        # (n, x, y, c) channels last to (n, c, x, y) channels first
        tensor = torch.from_numpy(np.ascontiguousarray(img_batch, dtype=np.float32))
        tensor = tensor.permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            predictions = self.model(tensor).cpu().numpy()

        if img_batch.shape[0] == 1:  # Single image
            return cast(NDArray[Any], predictions[0, :])

        return cast(NDArray[Any], predictions)


class VggFaceClassifier(nn.Module):  # type: ignore[misc]
    """
    Classifier built on top of the VGG-Face trunk, the way the age, gender and race
    models are built in the tensorflow backend. Its classifier is fed with the fc7
    features, so the 2622 way one of VGG-Face is left out.

    Args:
        classes (int): number of classes to predict
    """

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.backbone = VggFaceBaseModel(include_top=False)
        self.predictions = nn.Conv2d(4096, classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Find the class distribution of given images
        Args:
            x (Tensor): (n, 3, 224, 224) shaped input in BGR
        Returns:
            predictions (Tensor): (n, classes) shaped probabilities
        """
        x = self.backbone.embed(x)
        x = self.predictions(x[:, :, None, None])
        return torch.softmax(torch.flatten(x, start_dim=1), dim=1)
