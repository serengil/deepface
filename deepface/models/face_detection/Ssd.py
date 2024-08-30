from typing import List
import os
from enum import IntEnum
import gdown
import cv2
import numpy as np
from deepface.models.face_detection import OpenCv
from deepface.commons import folder_utils
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long, c-extension-no-member


class SsdClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> dict:
        """
        Build a ssd detector model
        Returns:
            model (dict)
        """

        home = folder_utils.get_deepface_home()

        # model structure
        output_model = os.path.join(home, ".deepface/weights/deploy.prototxt")
        if not os.path.isfile(output_model):
            logger.info(f"{os.path.basename(output_model)} will be downloaded...")
            url = "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt"
            gdown.download(url, output_model, quiet=False)

        # pre-trained weights
        output_weights = os.path.join(home, ".deepface/weights/res10_300x300_ssd_iter_140000.caffemodel")
        if not os.path.isfile(output_weights):
            logger.info(f"{os.path.basename(output_weights)} will be downloaded...")
            url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            gdown.download(url, output_weights, quiet=False)

        try:
            face_detector = cv2.dnn.readNetFromCaffe(output_model, output_weights)
        except Exception as err:
            raise ValueError(
                "Exception while calling opencv.dnn module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err

        return {
            "face_detector": face_detector,
            "opencv_module": OpenCv.OpenCvClient()
        }

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

        opencv_module: OpenCv.OpenCvClient = self.model["opencv_module"]

        target_size = (300, 300)

        original_size = img.shape

        current_img = cv2.resize(img, target_size)

        aspect_ratio_x = original_size[1] / target_size[1]
        aspect_ratio_y = original_size[0] / target_size[0]

        imageBlob = cv2.dnn.blobFromImage(image=current_img)

        face_detector = self.model["face_detector"]
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
        faces = faces[(faces[:, ssd_labels.is_face] == 1) & (faces[:, ssd_labels.confidence] >= 0.90)]
        margins = [ssd_labels.left, ssd_labels.top, ssd_labels.right, ssd_labels.bottom]
        faces[:, margins] = np.int32(faces[:, margins] * 300)
        faces[:, margins] = np.int32(faces[:, margins] * [aspect_ratio_x, aspect_ratio_y, aspect_ratio_x, aspect_ratio_y])
        faces[:, [ssd_labels.right, ssd_labels.bottom]] -= faces[:, [ssd_labels.left, ssd_labels.top]]

        resp = []
        for face in faces:
            confidence = float(face[2])
            x, y, w, h = map(int, face[margins])
            detected_face = img[y : y + h, x : x + w]

            left_eye, right_eye = opencv_module.find_eyes(detected_face)

            # eyes found in the detected face instead image itself
            # detected face's coordinates should be added
            if left_eye is not None:
                left_eye = x + int(left_eye[0]), y + int(left_eye[1])
            if right_eye is not None:
                right_eye = x + int(right_eye[0]), y + int(right_eye[1])

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            resp.append(facial_area)
        return resp
