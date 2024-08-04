# built-in dependencies
import os

# 3rd party dependencies
import gdown
import numpy as np

# project dependencies
from deepface.basemodels import VGGFace
from deepface.commons import package_utils, folder_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

logger = Logger()

# --------------------------
# pylint: disable=line-too-long
# --------------------------
# dependency configurations
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation
# --------------------------
# Labels for the ethnic phenotypes that can be detected by the model.
labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]

# pylint: disable=too-few-public-methods
class RaceClient(Demography):
    """
    Race model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Race"

    def predict(self, img: np.ndarray) -> np.ndarray:
        # model.predict causes memory issue when it is called in a for loop
        # return self.model.predict(img, verbose=0)[0, :]
        return self.model(img, training=False).numpy()[0, :]


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
) -> Model:
    """
    Construct race model, download its weights and load
    """

    model = VGGFace.base_model()

    # --------------------------

    classes = 6
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    race_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    home = folder_utils.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/race_model_single_batch.h5") != True:
        logger.info("race_model_single_batch.h5 will be downloaded...")

        output = home + "/.deepface/weights/race_model_single_batch.h5"
        gdown.download(url, output, quiet=False)

    race_model.load_weights(home + "/.deepface/weights/race_model_single_batch.h5")

    return race_model
