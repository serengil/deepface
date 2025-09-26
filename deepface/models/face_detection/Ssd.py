# built-in dependencies
from typing import List, Any
from enum import IntEnum

# 3rd party dependencies
import cv2
import numpy as np

# project dependencies
from deepface.models.face_detection import OpenCv
from deepface.commons import weight_utils
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long, c-extension-no-member

MODEL_URL = "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt"
WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


class SsdClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a ssd detector model
        Returns:
            model (Any): ssd model
        """

        # model structure
        output_model = weight_utils.download_weights_if_necessary(
            file_name="deploy.prototxt",
            source_url=MODEL_URL,
        )

        # pre-trained weights
        output_weights = weight_utils.download_weights_if_necessary(
            file_name="res10_300x300_ssd_iter_140000.caffemodel",
            source_url=WEIGHTS_URL,
        )

        try:
            face_detector = cv2.dnn.readNetFromCaffe(output_model, output_weights)
        except Exception as err:
            raise ValueError(
                "Exception while calling opencv.dnn module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err

        return face_detector

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with ssd

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """

        # Because cv2.dnn.blobFromImage expects CV_8U (8-bit unsigned integer) values
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        target_size = (300, 300)

        original_size = img.shape

        current_img = cv2.resize(img, target_size)

        aspect_ratio_x = original_size[1] / target_size[1]
        aspect_ratio_y = original_size[0] / target_size[0]

        imageBlob = cv2.dnn.blobFromImage(image=current_img)

        face_detector = self.model
        face_detector.setInput(imageBlob)
        detections = face_detector.forward()

        class ssd_labels(IntEnum):
            img_id = 0
            is_face = 1
            confidence = 2
            left = 3
            top = 4
            right = 5
            bottom = 6

        faces = detections[0][0]
        faces = faces[
            (faces[:, ssd_labels.is_face] == 1) & (faces[:, ssd_labels.confidence] >= 0.90)
        ]
        margins = [ssd_labels.left, ssd_labels.top, ssd_labels.right, ssd_labels.bottom]
        faces[:, margins] = np.int32(faces[:, margins] * 300)
        faces[:, margins] = np.int32(
            faces[:, margins] * [aspect_ratio_x, aspect_ratio_y, aspect_ratio_x, aspect_ratio_y]
        )
        faces[:, [ssd_labels.right, ssd_labels.bottom]] -= faces[
            :, [ssd_labels.left, ssd_labels.top]
        ]

        resp = []
        for face in faces:
            confidence = float(face[ssd_labels.confidence])
            x, y, w, h = map(int, face[margins])

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=None,
                right_eye=None,
                confidence=confidence,
            )
            resp.append(facial_area)
        return resp
