from typing import List
import os
import bz2
import gdown
import numpy as np
from deepface.commons import folder_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=too-few-public-methods


class DlibClient(FacialRecognition):
    """
    Dlib model class
    """

    def __init__(self):
        self.model = DlibResNet()
        self.model_name = "Dlib"
        self.input_shape = (150, 150)
        self.output_shape = 128

    def forward(self, img: np.ndarray) -> List[float]:
        """
        Find embeddings with Dlib model.
            This model necessitates the override of the forward method
            because it is not a keras model.
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # return self.model.predict(img)[0].tolist()

        # extract_faces returns 4 dimensional images
        if len(img.shape) == 4:
            img = img[0]

        # bgr to rgb
        img = img[:, :, ::-1]  # bgr to rgb

        # img is in scale of [0, 1] but expected [0, 255]
        if img.max() <= 1:
            img = img * 255

        img = img.astype(np.uint8)

        img_representation = self.model.model.compute_face_descriptor(img)
        img_representation = np.array(img_representation)
        img_representation = np.expand_dims(img_representation, axis=0)
        return img_representation[0].tolist()


class DlibResNet:
    def __init__(self):

        ## this is not a must dependency. do not import it in the global level.
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional dependency, ensure the library is installed."
                "Please install using 'pip install dlib' "
            ) from e

        home = folder_utils.get_deepface_home()
        filename = "dlib_face_recognition_resnet_model_v1.dat"
        weight_file = os.path.join(home, ".deepface/weights", filename)

        # download pre-trained model if it does not exist
        if not os.path.isfile(weight_file):
            logger.info(f"{filename} is going to be downloaded")
            url = f"http://dlib.net/files/{filename + '.bz2'}"
            output = weight_file + ".bz2"
            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            with open(weight_file, "wb") as f:
                f.write(data)

        self.model = dlib.face_recognition_model_v1(weight_file)

        # return None  # classes must return None
