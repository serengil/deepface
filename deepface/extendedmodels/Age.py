import os
import gdown
import numpy as np
import tensorflow as tf
from deepface.basemodels import VGGFace
from deepface.commons import functions

# ----------------------------------------
# dependency configurations

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

# ----------------------------------------


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5",
):

    model = VGGFace.baseModel()

    # --------------------------

    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    age_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    home = functions.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/age_model_weights.h5") != True:
        print("age_model_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/age_model_weights.h5"
        gdown.download(url, output, quiet=False)

    age_model.load_weights(home + "/.deepface/weights/age_model_weights.h5")

    return age_model

    # --------------------------


def findApparentAge(age_predictions):
    output_indexes = np.array(list(range(0, 101)))
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age
