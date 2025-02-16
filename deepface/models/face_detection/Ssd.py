# built-in dependencies
from typing import List, Union
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

    def build_model(self) -> dict:
        """
        Build a ssd detector model
        Returns:
            model (dict)
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

        return {"face_detector": face_detector, "opencv_module": OpenCv.OpenCvClient()}

    def detect_faces(
        self,
        img: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]:
        """
        Detect and align faces with ssd in a batch of images

        Args:
            img (Union[np.ndarray, List[np.ndarray]]): 
            Pre-loaded image as numpy array or a list of those

        Returns:
            results (Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]): 
            A list or a list of lists of FacialAreaRegion objects
        """
        if isinstance(img, np.ndarray):
            imgs = [img]
        else:
            imgs = img

        batch_results = []

        for single_img in imgs:
            # Because cv2.dnn.blobFromImage expects CV_8U (8-bit unsigned integer) values
            if single_img.dtype != np.uint8:
                single_img = single_img.astype(np.uint8)

            opencv_module: OpenCv.OpenCvClient = self.model["opencv_module"]

            target_size = (300, 300)
            original_size = single_img.shape
            current_img = cv2.resize(single_img, target_size)

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
                detected_face = single_img[y : y + h, x : x + w]

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

            batch_results.append(resp)

        return batch_results if len(batch_results) > 1 else batch_results[0]
