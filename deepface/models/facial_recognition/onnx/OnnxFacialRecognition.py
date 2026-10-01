# built-in dependencies
from typing import Any, List, Union, cast

# 3rd party dependencies
from numpy.typing import NDArray

# project dependencies
from deepface.commons.onnx_utils import run_session
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# Notice that facial recognition models with onnx graphs exported from pytorch ones
# should be inherited from this class


class OnnxFacialRecognition(FacialRecognition):
    """
    Base class of facial recognition models running on onnxruntime
    """

    model: Any  # onnxruntime.InferenceSession

    def predict(self, img: NDArray[Any]) -> NDArray[Any]:
        """
        Find raw representations of given images
        Args:
            img (np.ndarray): pre-loaded image in BGR with (X, X, 3) or (n, X, X, 3) shape
        Returns:
            embeddings (np.ndarray): (n, output_shape) shaped representations
        """
        return run_session(self.model, img)

    def forward(self, img: NDArray[Any]) -> Union[List[float], List[List[float]]]:
        embeddings = self.predict(img)

        if embeddings.shape[0] == 1:
            return cast(List[float], embeddings[0].tolist())
        return cast(List[List[float]], embeddings.tolist())
