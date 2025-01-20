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

# -------------------------------------
# pylint: disable=line-too-long
# -------------------------------------
# dependency configurations

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

WEIGHTS_URL="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5"

# Labels for the genders that can be detected by the model.
labels = ["Woman", "Man"]

# pylint: disable=too-few-public-methods
class GenderClient(Demography):
    """
    Gender model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Gender"

    def predict(self, img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Predict gender probabilities for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (n, 2)
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Prediction
        predictions = self._predict_internal(imgs)

        return predictions

def load_model(
    url=WEIGHTS_URL,
) -> Model:
    """
    Construct gender model, download its weights and load
    Returns:
        model (Model)
    """

    model = VGGFace.base_model()

    # --------------------------

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    gender_model = Model(inputs=model.inputs, outputs=base_model_output)

    # --------------------------

    # load weights
    weight_file = weight_utils.download_weights_if_necessary(
        file_name="gender_model_weights.h5", source_url=url
    )

    gender_model = weight_utils.load_model_weights(
        model=gender_model, weight_file=weight_file
    )

    return gender_model
