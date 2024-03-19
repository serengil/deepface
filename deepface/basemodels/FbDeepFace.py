from typing import List
import os
import zipfile
import gdown
import numpy as np
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.models.FacialRecognition import FacialRecognition

logger = Logger(module="basemodels.FbDeepFace")

# --------------------------------
# dependency configuration

tf_major = package_utils.get_tf_major_version()
tf_minor = package_utils.get_tf_minor_version()

if tf_major == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        MaxPooling2D,
        Flatten,
        Dense,
        Dropout,
        LocallyConnected2D,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        MaxPooling2D,
        Flatten,
        Dense,
        Dropout,
        LocallyConnected2D,
    )


# -------------------------------------
# pylint: disable=line-too-long, too-few-public-methods
class DeepFaceClient(FacialRecognition):
    """
    Fb's DeepFace model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "DeepFace"
        self.input_shape = (152, 152)
        self.output_shape = 4096

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """
        find embeddings with OpenFace model
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()
        return self.model(img, training=False).numpy()[0].tolist()


def load_model(
    url="https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip",
) -> Model:
    """
    Construct DeepFace model, download its weights and load
    """
    base_model = Sequential()
    base_model.add(
        Convolution2D(32, (11, 11), activation="relu", name="C1", input_shape=(152, 152, 3))
    )
    base_model.add(MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"))
    base_model.add(Convolution2D(16, (9, 9), activation="relu", name="C3"))
    base_model.add(LocallyConnected2D(16, (9, 9), activation="relu", name="L4"))
    base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation="relu", name="L5"))
    base_model.add(LocallyConnected2D(16, (5, 5), activation="relu", name="L6"))
    base_model.add(Flatten(name="F0"))
    base_model.add(Dense(4096, activation="relu", name="F7"))
    base_model.add(Dropout(rate=0.5, name="D0"))
    base_model.add(Dense(8631, activation="softmax", name="F8"))

    # ---------------------------------

    home = folder_utils.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5") != True:
        logger.info("VGGFace2_DeepFace_weights_val-0.9034.h5 will be downloaded...")

        output = home + "/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5.zip"

        gdown.download(url, output, quiet=False)

        # unzip VGGFace2_DeepFace_weights_val-0.9034.h5.zip
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(home + "/.deepface/weights/")

    base_model.load_weights(home + "/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5")

    # drop F8 and D0. F7 is the representation layer.
    deepface_model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)

    return deepface_model
