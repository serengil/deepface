import os
from pathlib import Path
from keras.models import Model, Sequential
from keras.layers import (
    Input,
    Convolution2D,
    ZeroPadding2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Activation,
)
import gdown

# ---------------------------------------


def get_base_model():
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


def loadModel(model_path=""):
    """
        Args:
            model_path: str
                If provided, this path will be used to load the model from.
    """
    if model_path:
        assert Path(model_path).exists()
        assert model_path.endswith(".h5")
    else:
        home = Path.home().as_posix()
        model_path = os.path.join(home, ".deepface/weights/vgg_face_weights.h5")
        if not os.path.isfile(model_path):
            print(f"vgg_face_weights.h5 will be downloaded to {model_path}")

            url = "https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo"
            gdown.download(url, model_path, quiet=False)

    # -----------------------------------

    print(f"Loading model from {model_path}")
    model = get_base_model()
    model.load_weights(model_path)

    # -----------------------------------

    # TO-DO: why?
    vgg_face_descriptor = Model(
        inputs=model.layers[0].input, outputs=model.layers[-2].output
    )

    return vgg_face_descriptor
