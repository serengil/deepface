import os
import gdown
import tensorflow as tf
from deepface.basemodels import Facenet
from deepface.commons import functions
from deepface.commons.logger import Logger

logger = Logger(module="basemodels.Facenet512")

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model
else:
    from tensorflow.keras.models import Model


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
) -> Model:

    model = Facenet.InceptionResNetV2(dimension=512)

    # -------------------------

    home = functions.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/facenet512_weights.h5") != True:
        logger.info("facenet512_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/facenet512_weights.h5"
        gdown.download(url, output, quiet=False)

    # -------------------------

    model.load_weights(home + "/.deepface/weights/facenet512_weights.h5")

    # -------------------------

    return model
