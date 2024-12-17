# stdlib dependencies
from typing import List, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.models.facial_recognition import VGGFace
from deepface.commons import package_utils, weight_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

logger = Logger()

# ----------------------------------------
# dependency configurations

tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

# ----------------------------------------

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5"
)

# pylint: disable=too-few-public-methods
class ApparentAgeClient(Demography):
    """
    Age model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Age"

    def predict(self, img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Predict apparent age(s) for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (n,)
        """
        # Convert to numpy array if input is list
        if isinstance(img, list):
            imgs = np.array(img)
        else:
            imgs = img

        # Remove batch dimension if exists
        imgs = imgs.squeeze()

        # Check input dimension
        if len(imgs.shape) == 3:
            # Single image - add batch dimension
            imgs = np.expand_dims(imgs, axis=0)

        # Batch prediction
        age_predictions = self.model.predict_on_batch(imgs)
        
        # Calculate apparent ages
        apparent_ages = np.array(
            [find_apparent_age(age_prediction) for age_prediction in age_predictions]
        )

        return apparent_ages


def load_model(
    url=WEIGHTS_URL,
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


def find_apparent_age(age_predictions: np.ndarray) -> np.float64:
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (?)
    Returns:
        apparent_age (float)
    """
    output_indexes = np.arange(0, 101)
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age
