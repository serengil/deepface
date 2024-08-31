# built-in dependencies
from typing import Any, List

# 3rd party dependencies
import numpy as np
import cv2 as cv

# project dependencies
from deepface.commons import weight_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

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

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="face_recognition_sface_2021dec.onnx", source_url=url
    )

    model = SFaceWrapper(model_path=weight_file)

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
