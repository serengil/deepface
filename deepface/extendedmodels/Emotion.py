# built-in dependencies
import os

# 3rd party dependencies
import gdown
import numpy as np
import cv2

# project dependencies
from deepface.commons import package_utils, folder_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

logger = Logger()

# -------------------------------------------
# pylint: disable=line-too-long
# -------------------------------------------
# dependency configuration
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
else:
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

# pylint: disable=too-few-public-methods
class EmotionClient(Demography):
    """
    Emotion model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Emotion"

    def predict(self, img: np.ndarray) -> np.ndarray:
        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = np.expand_dims(img_gray, axis=0)

        emotion_predictions = self.model.predict(img_gray, verbose=0)[0, :]
        return emotion_predictions


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
) -> Sequential:
    """
    Consruct emotion model, download and load weights
    """

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

    home = folder_utils.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/facial_expression_model_weights.h5") != True:
        logger.info("facial_expression_model_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/facial_expression_model_weights.h5"
        gdown.download(url, output, quiet=False)

    model.load_weights(home + "/.deepface/weights/facial_expression_model_weights.h5")

    return model
