# built-in dependencies
from typing import Any, List, TypeVar, Union, cast

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

# project dependencies
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

TorchModel = TypeVar("TorchModel", bound=nn.Module)

# Notice that all facial recognition models with pytorch backend must be
# inherited from this class


class TorchFacialRecognition(FacialRecognition):
    """
    Base class of facial recognition models having pytorch backend.

    Inputs are still expected to be channels last - (n, X, X, 3) shaped and in BGR -
    as in the tensorflow backend, they are transposed to channels first internally.
    """

    model: nn.Module
    device: torch.device

    def to_tensor(self, img: NDArray[Any]) -> torch.Tensor:
        """
        Convert a channels last numpy image into a channels first tensor
        Args:
            img (np.ndarray): pre-loaded image in BGR with (X, X, 3) or (n, X, X, 3) shape
        Returns:
            tensor (torch.Tensor): (n, 3, X, X) shaped tensor on the model's device
        """
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        if img.ndim != 4 or img.shape[0] < 1:
            raise ValueError(f"Input image must be (n, X, X, 3) shaped but it is {img.shape}")

        tensor = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
        return cast(torch.Tensor, tensor.permute(0, 3, 1, 2).to(self.device))

    def predict(self, img: NDArray[Any]) -> NDArray[Any]:
        """
        Find raw representations of given images
        Args:
            img (np.ndarray): pre-loaded image in BGR with (X, X, 3) or (n, X, X, 3) shape
        Returns:
            embeddings (np.ndarray): (n, output_shape) shaped representations
        """
        with torch.no_grad():
            embeddings = self.model(self.to_tensor(img)).cpu().numpy()
        return cast(NDArray[Any], embeddings)

    def forward(self, img: NDArray[Any]) -> Union[List[float], List[List[float]]]:
        embeddings = self.predict(img)

        if embeddings.shape[0] == 1:
            return cast(List[float], embeddings[0].tolist())
        return cast(List[List[float]], embeddings.tolist())


def load_model_weights(model: TorchModel, weight_file: str) -> TorchModel:
    """
    Load pre-trained weights for a given pytorch model
    Args:
        model (TorchModel): pre-built model
        weight_file (str): exact path of pre-trained weights
    Returns:
        model (TorchModel): pre-built model with updated weights
    """
    try:
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    except Exception as err:
        raise ValueError(
            f"An exception occurred while loading the pre-trained weights from {weight_file}."
            "This might have happened due to an interruption during the download."
            "You may want to delete it and allow DeepFace to download it again during the next run."
            "If the issue persists, consider downloading the file directly from the source "
            "and copying it to the target folder."
        ) from err

    model.load_state_dict(state_dict)
    model.eval()
    return model
