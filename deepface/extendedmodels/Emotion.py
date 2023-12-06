import os
import gdown
import tensorflow as tf
from deepface.commons import functions
from deepface.commons.logger import Logger

logger = Logger()

# -------------------------------------------
# pylint: disable=line-too-long
# -------------------------------------------
# dependency configuration
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
elif tf_version == 2:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
    )
# -------------------------------------------

# Labels for the emotions that can be detected by the model.
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
):

    num_classes = 7

    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    # ----------------------------

    home = functions.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/facial_expression_model_weights.h5") != True:
        logger.info("facial_expression_model_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/facial_expression_model_weights.h5"
        gdown.download(url, output, quiet=False)

    model.load_weights(home + "/.deepface/weights/facial_expression_model_weights.h5")

    return model
