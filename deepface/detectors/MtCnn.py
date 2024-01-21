from typing import List, Tuple
import cv2
import numpy as np
from mtcnn import MTCNN
from deepface.models.Detector import Detector
from deepface.modules import detection

# pylint: disable=too-few-public-methods
class MtCnnClient(Detector):
    """
    Class to cover common face detection functionalitiy for MtCnn backend
    """

    def __init__(self):
        self.model = MTCNN()

    def detect_faces(
        self, img: np.ndarray, align: bool = True
    ) -> List[Tuple[np.ndarray, List[float], float]]:
        """
        Detect and align face with mtcnn
        Args:
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            results (List[Tuple[np.ndarray, List[float], float]]): A list of tuples
                where each tuple contains:
                - detected_face (np.ndarray): The detected face as a NumPy array.
                - face_region (List[float]): The image region represented as
                    a list of floats e.g. [x, y, w, h]
                - confidence (float): The confidence score associated with the detected face.

        Example:
            results = [
                (array(..., dtype=uint8), [110, 60, 150, 380], 0.99),
                (array(..., dtype=uint8), [150, 50, 299, 375], 0.98),
                (array(..., dtype=uint8), [120, 55, 300, 371], 0.96),
            ]
        """

        resp = []

        detected_face = None
        img_region = [0, 0, img.shape[1], img.shape[0]]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
        detections = self.model.detect_faces(img_rgb)

        if len(detections) > 0:

            for current_detection in detections:
                x, y, w, h = current_detection["box"]
                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                img_region = [x, y, w, h]
                confidence = current_detection["confidence"]

                if align:
                    keypoints = current_detection["keypoints"]
                    left_eye = keypoints["left_eye"]
                    right_eye = keypoints["right_eye"]
                    detected_face = detection.align_face(
                        img=detected_face, left_eye=left_eye, right_eye=right_eye
                    )

                resp.append((detected_face, img_region, confidence))

        return resp
