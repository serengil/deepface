# stdlib dependencies
from typing import List, Union, Any, cast

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.models.facial_recognition import VGGFace
from deepface.commons import package_utils, weight_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

logger = Logger()

# dependency configurations

tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5"
)


# pylint: disable=too-few-public-methods
class ApparentAgeClient(Demography):
    """
    Age model class
    """

    def __init__(self) -> None:
        self.model = load_model()
        self.model_name = "Age"

    def predict(
        self, img: Union[NDArray[Any], List[NDArray[Any]]]
    ) -> Union[np.float64, NDArray[Any]]:
        """
        Predict apparent age(s) for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (age_classes,) if single image,
            np.ndarray (n, age_classes) if batched images.
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Prediction from 3 channels image
        age_predictions = self._predict_internal(imgs)

        # Calculate apparent ages
        if len(age_predictions.shape) == 1:  # Single prediction list
            return find_apparent_age(age_predictions)

        return np.array([find_apparent_age(age_prediction) for age_prediction in age_predictions])


def load_model(
    url: str = WEIGHTS_URL,
) -> Model:
    """
    Construct age model, download its weights and load
    Returns:
        model (Model)
    """

    model = VGGFace.base_model()

    # --------------------------

    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    age_model = Model(inputs=model.inputs, outputs=base_model_output)

    # --------------------------

    # load weights
    weight_file = weight_utils.download_weights_if_necessary(
        file_name="age_model_weights.h5", source_url=url
    )

    age_model = weight_utils.load_model_weights(model=age_model, weight_file=weight_file)

    return age_model


def find_apparent_age(age_predictions: NDArray[Any]) -> np.float64:
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (age_classes,)
    Returns:
        apparent_age (float)
    """
    assert (
        len(age_predictions.shape) == 1
    ), f"Input should be a list of predictions, not batched. Got shape: {age_predictions.shape}"
    output_indexes = np.arange(0, 101)
    apparent_age = cast(np.float64, np.sum(age_predictions * output_indexes))
    return apparent_age
