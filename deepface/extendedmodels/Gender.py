import os
import gdown
import tensorflow as tf
import numpy as np
from deepface.basemodels import VGGFace
from deepface.commons import functions
from deepface.commons.logger import Logger
from deepface.models.Demography import Demography

logger = Logger(module="extendedmodels.Gender")

# -------------------------------------
# pylint: disable=line-too-long
# -------------------------------------
# dependency configurations

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation
# -------------------------------------

# Labels for the genders that can be detected by the model.
labels = ["Woman", "Man"]

# pylint: disable=too-few-public-methods
class Gender(Demography):
    """
    Gender model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Gender"

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict(img, verbose=0)[0, :]


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
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

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    home = functions.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/gender_model_weights.h5") != True:
        logger.info("gender_model_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/gender_model_weights.h5"
        gdown.download(url, output, quiet=False)

    gender_model.load_weights(home + "/.deepface/weights/gender_model_weights.h5")

    return gender_model
