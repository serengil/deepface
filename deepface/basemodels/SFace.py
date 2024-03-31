import os
from typing import Any, List

import numpy as np
import cv2 as cv
import gdown

from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.models.FacialRecognition import FacialRecognition

logger = Logger(module="basemodels.SFace")

# pylint: disable=line-too-long, too-few-public-methods


class SFaceClient(FacialRecognition):
    """
    SFace model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "SFace"
        self.input_shape = (112, 112)
        self.output_shape = 128

    def forward(self, img: np.ndarray) -> List[float]:
        """
        Find embeddings with SFace model
            This model necessitates the override of the forward method
            because it is not a keras model.
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # return self.model.predict(img)[0].tolist()

        # revert the image to original format and preprocess using the model
        input_blob = (img[0] * 255).astype(np.uint8)

        embeddings = self.model.model.feature(input_blob)

        return embeddings[0].tolist()


def load_model(
    url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
) -> Any:
    """
    Construct SFace model, download its weights and load
    """

    home = folder_utils.get_deepface_home()

    file_name = home + "/.deepface/weights/face_recognition_sface_2021dec.onnx"

    if not os.path.isfile(file_name):

        logger.info("sface weights will be downloaded...")

        gdown.download(url, file_name, quiet=False)

    model = SFaceWrapper(model_path=file_name)

    return model


class SFaceWrapper:
    def __init__(self, model_path):
        """
        SFace wrapper covering model construction, layer infos and predict
        """
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
