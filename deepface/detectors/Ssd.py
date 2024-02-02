from typing import List
import os
import gdown
import cv2
import pandas as pd
import numpy as np
from deepface.detectors import OpenCv
from deepface.commons import folder_utils
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.modules import detection
from deepface.commons.logger import Logger

logger = Logger(module="detectors.SsdWrapper")

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
        if os.path.isfile(home + "/.deepface/weights/deploy.prototxt") != True:

            logger.info("deploy.prototxt will be downloaded...")

            url = "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt"

            output = home + "/.deepface/weights/deploy.prototxt"

            gdown.download(url, output, quiet=False)

        # pre-trained weights
        if (
            os.path.isfile(home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel")
            != True
        ):

            logger.info("res10_300x300_ssd_iter_140000.caffemodel will be downloaded...")

            url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

            output = home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"

            gdown.download(url, output, quiet=False)

        try:
            face_detector = cv2.dnn.readNetFromCaffe(
                home + "/.deepface/weights/deploy.prototxt",
                home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel",
            )
        except Exception as err:
            raise ValueError(
                "Exception while calling opencv.dnn module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err

        detector = {}
        detector["face_detector"] = face_detector
        detector["opencv_module"] = OpenCv.OpenCvClient()

        return detector

    def detect_faces(
        self, img: np.ndarray, align: bool = True, expand_percentage: int = 0
    ) -> List[DetectedFace]:
        """
        Detect and align face with ssd

        Args:
            img (np.ndarray): pre-loaded image as numpy array

            align (bool): flag to enable or disable alignment after detection (default is True)

            expand_percentage (int): expand detected facial area with a percentage

        Returns:
            results (List[Tuple[DetectedFace]): A list of DetectedFace objects
                where each object contains:

            - img (np.ndarray): The detected face as a NumPy array.

            - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h

            - confidence (float): The confidence score associated with the detected face.
        """
        resp = []

        detected_face = None

        ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

        target_size = (300, 300)

        original_size = img.shape

        current_img = cv2.resize(img, target_size)

        aspect_ratio_x = original_size[1] / target_size[1]
        aspect_ratio_y = original_size[0] / target_size[0]

        imageBlob = cv2.dnn.blobFromImage(image=current_img)

        face_detector = self.model["face_detector"]
        face_detector.setInput(imageBlob)
        detections = face_detector.forward()

        detections_df = pd.DataFrame(detections[0][0], columns=ssd_labels)

        detections_df = detections_df[detections_df["is_face"] == 1]  # 0: background, 1: face
        detections_df = detections_df[detections_df["confidence"] >= 0.90]

        detections_df["left"] = (detections_df["left"] * 300).astype(int)
        detections_df["bottom"] = (detections_df["bottom"] * 300).astype(int)
        detections_df["right"] = (detections_df["right"] * 300).astype(int)
        detections_df["top"] = (detections_df["top"] * 300).astype(int)

        if detections_df.shape[0] > 0:

            for _, instance in detections_df.iterrows():

                left = instance["left"]
                right = instance["right"]
                bottom = instance["bottom"]
                top = instance["top"]

                x = int(left * aspect_ratio_x)
                y = int(top * aspect_ratio_y)
                w = int(right * aspect_ratio_x) - int(left * aspect_ratio_x)
                h = int(bottom * aspect_ratio_y) - int(top * aspect_ratio_y)

                # expand the facial area to be extracted and stay within img.shape limits
                x2 = max(0, x - int((w * expand_percentage) / 100))  # expand left
                y2 = max(0, y - int((h * expand_percentage) / 100))  # expand top
                w2 = min(img.shape[1], w + int((w * expand_percentage) / 100))  # expand right
                h2 = min(img.shape[0], h + int((h * expand_percentage) / 100))  # expand bottom

                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                detected_face = img[int(y2) : int(y2 + h2), int(x2) : int(x2 + w2)]

                face_region = FacialAreaRegion(x=x, y=y, w=w, h=h)

                confidence = instance["confidence"]

                if align:
                    opencv_module: OpenCv.OpenCvClient = self.model["opencv_module"]
                    left_eye, right_eye = opencv_module.find_eyes(detected_face)
                    detected_face = detection.align_face(
                        img=detected_face, left_eye=left_eye, right_eye=right_eye
                    )

                detected_face_obj = DetectedFace(
                    img=detected_face,
                    facial_area=face_region,
                    confidence=confidence,
                )

                resp.append(detected_face_obj)
        return resp
