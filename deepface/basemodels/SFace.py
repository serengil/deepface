import os
from typing import Any

import numpy as np
import cv2 as cv
import gdown

from deepface.commons import functions
from deepface.commons.logger import Logger

logger = Logger(module="basemodels.SFace")

# pylint: disable=line-too-long, too-few-public-methods


class _Layer:
    input_shape = (None, 112, 112, 3)
    output_shape = (None, 1, 128)


class SFaceModel:
    def __init__(self, model_path):

        try:
            self.model = cv.FaceRecognizerSF.create(
                model=model_path, config="", backend_id=0, target_id=0
            )
        except Exception as err:
            raise ValueError(
                "Exception while calling opencv.FaceRecognizerSF module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err

        self.layers = [_Layer()]

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Preprocess
        input_blob = (image[0] * 255).astype(
            np.uint8
        )  # revert the image to original format and preprocess using the model

        # Forward
        embeddings = self.model.feature(input_blob)

        return embeddings


def load_model(
    url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
) -> Any:

    home = functions.get_deepface_home()

    file_name = home + "/.deepface/weights/face_recognition_sface_2021dec.onnx"

    if not os.path.isfile(file_name):

        logger.info("sface weights will be downloaded...")

        gdown.download(url, file_name, quiet=False)

    model = SFaceModel(model_path=file_name)

    return model
