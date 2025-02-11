# built-in dependencies
from typing import List, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.commons import weight_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=too-few-public-methods
WEIGHT_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"


class DlibClient(FacialRecognition):
    """
    Dlib model class
    """

    def __init__(self):
        self.model = DlibResNet()
        self.model_name = "Dlib"
        self.input_shape = (150, 150)
        self.output_shape = 128

    def forward(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        """
        Find embeddings with Dlib model.
            This model necessitates the override of the forward method
            because it is not a keras model.
        Args:
            img (np.ndarray): pre-loaded image(s) in BGR
        Returns
            embeddings (list of lists or list of floats): multi-dimensional vectors
        """
        # Handle single image case
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)

        # bgr to rgb
        img = img[:, :, :, ::-1]  # bgr to rgb

        # img is in scale of [0, 1] but expected [0, 255]
        if img.max() <= 1:
            img = img * 255

        img = img.astype(np.uint8)

        embeddings = self.model.model.compute_face_descriptor(img)
        embeddings = [np.array(embedding).tolist() for embedding in embeddings]
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return embeddings


class DlibResNet:
    def __init__(self):

        # This is not a must dependency. Don't import it in the global level.
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional dependency, ensure the library is installed."
                "Please install using 'pip install dlib' "
            ) from e

        weight_file = weight_utils.download_weights_if_necessary(
            file_name="dlib_face_recognition_resnet_model_v1.dat",
            source_url=WEIGHT_URL,
            compress_type="bz2",
        )

        self.model = dlib.face_recognition_model_v1(weight_file)

        # return None  # classes must return None
