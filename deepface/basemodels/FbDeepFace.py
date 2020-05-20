import os
from pathlib import Path
import gdown
import keras
from keras.models import Model, Sequential
from keras.layers import (
    Convolution2D,
    LocallyConnected2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
import zipfile

# -------------------------------------


def get_base_model():
    base_model = Sequential()
    base_model.add(
        Convolution2D(
            32, (11, 11), activation="relu", name="C1", input_shape=(152, 152, 3)
        )
    )
    base_model.add(MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"))
    base_model.add(Convolution2D(16, (9, 9), activation="relu", name="C3"))
    base_model.add(LocallyConnected2D(16, (9, 9), activation="relu", name="L4"))
    base_model.add(
        LocallyConnected2D(16, (7, 7), strides=2, activation="relu", name="L5")
    )
    base_model.add(LocallyConnected2D(16, (5, 5), activation="relu", name="L6"))
    base_model.add(Flatten(name="F0"))
    base_model.add(Dense(4096, activation="relu", name="F7"))
    base_model.add(Dropout(rate=0.5, name="D0"))
    base_model.add(Dense(8631, activation="softmax", name="F8"))
    return base_model


def loadModel(model_path=""):
    # ---------------------------------
    if model_path:
        assert Path(model_path).exists()
        assert model_path.endswith(".h5")
    else:
        home = Path.home().as_posix()
        model_path = os.path.join(
            home, ".deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5"
        )
        if not os.path.isfile(model_path):
            print("VGGFace2_DeepFace_weights_val-0.9034.h5 will be downloaded...")

            url = "https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip"

            zip_path = os.path.join(
                home, ".deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5.zip"
            )
            gdown.download(url, zip_path, quiet=False)

            # unzip VGGFace2_DeepFace_weights_val-0.9034.h5.zip
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(home, "/.deepface/weights/"))

    print(f"Loading model from {model_path}")
    base_model = get_base_model()
    base_model.load_weights(model_path)

    # drop F8 and D0. F7 is the representation layer.
    deepface_model = Model(
        inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output
    )

    return deepface_model
