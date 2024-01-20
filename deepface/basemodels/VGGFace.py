import os
import gdown
import tensorflow as tf
from deepface.commons import functions
from deepface.commons.logger import Logger
from deepface.models.FacialRecognition import FacialRecognition

logger = Logger(module="basemodels.VGGFace")

# ---------------------------------------

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
        Lambda,
    )
    from keras import backend as K
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
        Lambda,
    )
    from tensorflow.keras import backend as K

# ---------------------------------------

# pylint: disable=too-few-public-methods
class VggFace(FacialRecognition):
    """
    VGG-Face model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "VGG-Face"


def base_model() -> Sequential:
    """
    Base model of VGG-Face being used for classification - not to find embeddings
    Returns:
        model (Sequential): model was trained to classify 2622 identities
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5",
) -> Model:
    """
    Final VGG-Face model being used for finding embeddings
    Returns:
        model (Model): returning 4096 dimensional vectors
    """

    model = base_model()

    home = functions.get_deepface_home()
    output = home + "/.deepface/weights/vgg_face_weights.h5"

    if os.path.isfile(output) != True:
        logger.info("vgg_face_weights.h5 will be downloaded...")
        gdown.download(url, output, quiet=False)

    model.load_weights(output)

    # 2622d dimensional model
    # vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    # 4096 dimensional model offers 6% to 14% increasement on accuracy!
    # - softmax causes underfitting
    # - added normalization layer to avoid underfitting with euclidean
    # as described here: https://github.com/serengil/deepface/issues/944
    base_model_output = Sequential()
    base_model_output = Flatten()(model.layers[-5].output)
    base_model_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name="norm_layer")(
        base_model_output
    )
    vgg_face_descriptor = Model(inputs=model.input, outputs=base_model_output)

    return vgg_face_descriptor
