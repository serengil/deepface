import os

import numpy as np
import cv2 as cv
import gdown

from deepface.commons import functions

class _Layer:
    input_shape = (None, 112, 112, 3)
    output_shape = (None, 1, 128)

class SFace:
    def __init__(self, model_path, backend_id=0, target_id=0):
        self._modelPath = model_path
        self._backendId = backend_id
        self._targetId = target_id
        self._model = cv.FaceRecognizerSF.create(
            model=self._modelPath,
            config="",
            backend_id=self._backendId,
            target_id=self._targetId)

        self.layers = [_Layer()]

    def _preprocess(self, image, bbox):
        if bbox is None:
            return image
        else:
            return self._model.alignCrop(image, bbox)

    def predict(self, image, bbox=None, **kwargs):
        # Preprocess
        image = (image[0] * 255).astype(np.uint8)  # revert the image to original format and preprocess using the model
        input_blob = self._preprocess(image, bbox)

        # Forward
        features = self._model.feature(input_blob)
        return features


def load_model(url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx", *args, **kwargs):
    home = functions.get_deepface_home()

    file_name = home + '/.deepface/weights/face_recognition_sface_2021dec.onnx'
    if not os.path.isfile(file_name):
        print("sface weights will be downloaded...")

        output = file_name
        gdown.download(url, output, quiet=False)

    model = SFace(file_name, 0, 0)
    return model
