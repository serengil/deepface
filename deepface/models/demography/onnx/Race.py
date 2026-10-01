# stdlib dependencies
from typing import Any, List, Union

# 3rd party dependencies
from numpy.typing import NDArray

# project dependencies
from deepface.models.demography.DemographyUtils import RACE_LABELS
from deepface.models.demography.onnx.OnnxDemography import OnnxDemography
from deepface.commons.onnx_utils import build_session
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long

WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/race_model_single_batch.onnx",
]

# Labels for the ethnic phenotypes that can be detected by the model.
labels = RACE_LABELS


# pylint: disable=too-few-public-methods
class RaceClient(OnnxDemography):
    """
    Race model class - onnx backend
    """

    def __init__(self) -> None:
        self.model_name = "Race"
        self.model = load_model()

    def predict(self, img: Union[NDArray[Any], List[NDArray[Any]]]) -> NDArray[Any]:
        """
        Predict race probabilities for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (n, n_races)
            where n_races is the number of race categories
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Prediction
        predictions = self._predict_internal(imgs)

        return predictions


def load_model(  # pylint: disable=dangerous-default-value
    url: Union[str, List[str]] = WEIGHTS_URL,
) -> Any:
    """
    Download race model's onnx graph if necessary and load it
    Returns:
        model (onnxruntime.InferenceSession)
    """
    return build_session(file_name="race_model_single_batch.onnx", source_url=url)
