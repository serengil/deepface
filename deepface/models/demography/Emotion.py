# stdlib dependencies
from typing import List, Union

# 3rd party dependencies
import numpy as np
import cv2

# project dependencies
from deepface.commons import package_utils, weight_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

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

# Labels for the emotions that can be detected by the model.
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

logger = Logger()

# pylint: disable=line-too-long, disable=too-few-public-methods

WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"


class EmotionClient(Demography):
    """
    Emotion model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Emotion"

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess single image for emotion detection
        Args:
            img: Input image (224, 224, 3)
        Returns:
            Preprocessed grayscale image (48, 48)
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        return img_gray

    def predict(self, img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Predict emotion probabilities for single or multiple faces
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            np.ndarray (n, n_emotions)
            where n_emotions is the number of emotion categories
        """
        # Preprocessing input image or image list.
        imgs = self._preprocess_batch_or_single_input(img)

        # Preprocess each image and add channel dimension for grayscale images
        processed_imgs = np.expand_dims(np.array([self._preprocess_image(img) for img in imgs]), axis=-1)

        # Reshape input for model (expected shape=(n, 48, 48, 1)), where n is the batch size
        processed_imgs = processed_imgs.reshape(processed_imgs.shape[0], 48, 48, 1)

        # Prediction
        # Emotion model input shape is (48, 48, 1, n), where n is the batch size
        predictions = self._predict_internal(processed_imgs)

        return predictions


def load_model(
    url=WEIGHTS_URL,
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

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="facial_expression_model_weights.h5", source_url=url
    )

    model = weight_utils.load_model_weights(model=model, weight_file=weight_file)

    return model
