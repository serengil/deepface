# built-in dependencies
import os
import pytest

# project dependencies
from deepface.commons import folder_utils, weight_utils, package_utils
from deepface.commons.logger import Logger

logger = Logger()

tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import (
        Dropout,
        Dense,
    )
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dropout,
        Dense,
    )


def test_loading_broken_weights():
    home = folder_utils.get_deepface_home()
    weight_file = os.path.join(home, ".deepface/weights/vgg_face_weights.h5")

    # construct a dummy model
    model = Sequential()

    # Add layers to the model
    model.add(
        Dense(units=64, activation="relu", input_shape=(100,))
    )  # Input layer with 100 features
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(Dense(units=32, activation="relu"))  # Hidden layer
    model.add(Dense(units=10, activation="softmax"))  # Output layer with 10 classes

    # vgg's weights cannot be loaded to this model
    with pytest.raises(
        ValueError,
        match="An exception occurred while loading the pre-trained weights from"
    ):
        model = weight_utils.load_model_weights(model=model, weight_file=weight_file)

    logger.info("✅ test loading broken weight file is done")
