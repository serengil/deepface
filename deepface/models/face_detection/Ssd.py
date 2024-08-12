from typing import List
import os
import gdown
import cv2
import pandas as pd
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

        detector = {}
        detector["face_detector"] = face_detector
        detector["opencv_module"] = OpenCv.OpenCvClient()

        return detector

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with ssd

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        opencv_module: OpenCv.OpenCvClient = self.model["opencv_module"]

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
                confidence = instance["confidence"]

                x = int(left * aspect_ratio_x)
                y = int(top * aspect_ratio_y)
                w = int(right * aspect_ratio_x) - int(left * aspect_ratio_x)
                h = int(bottom * aspect_ratio_y) - int(top * aspect_ratio_y)

                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

                left_eye, right_eye = opencv_module.find_eyes(detected_face)

                # eyes found in the detected face instead image itself
                # detected face's coordinates should be added
                if left_eye is not None:
                    left_eye = (int(x + left_eye[0]), int(y + left_eye[1]))
                if right_eye is not None:
                    right_eye = (int(x + right_eye[0]), int(y + right_eye[1]))

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
