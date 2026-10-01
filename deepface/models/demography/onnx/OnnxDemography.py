# built-in dependencies
from typing import Any, cast

# 3rd party dependencies
from numpy.typing import NDArray

# project dependencies
from deepface.models.Demography import Demography
from deepface.commons.onnx_utils import run_session
from deepface.commons.logger import Logger

logger = Logger()

# Notice that facial attribute analysis models with onnx graphs exported from pytorch
# ones should be inherited from this class


# pylint: disable=too-few-public-methods
class OnnxDemography(Demography):
    """
    Base class of facial attribute analysis models running on onnxruntime
    """

    model: Any  # onnxruntime.InferenceSession

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

        predictions = run_session(self.model, img_batch)

        if img_batch.shape[0] == 1:  # Single image
            return cast(NDArray[Any], predictions[0, :])

        return predictions
