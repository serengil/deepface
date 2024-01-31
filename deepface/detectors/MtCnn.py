from typing import List
import numpy as np
from mtcnn import MTCNN
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.modules import detection

# pylint: disable=too-few-public-methods
class MtCnnClient(Detector):
    """
    Class to cover common face detection functionalitiy for MtCnn backend
    """

    def __init__(self):
        self.model = MTCNN()

    def detect_faces(self, img: np.ndarray, align: bool = True) -> List[DetectedFace]:
        """
        Detect and align face with mtcnn
        Args:
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            results (List[DetectedFace]): A list of DetectedFace objects
                where each object contains:
            - img (np.ndarray): The detected face as a NumPy array.
            - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h
            - confidence (float): The confidence score associated with the detected face.
        """

        resp = []

        detected_face = None

        # mtcnn expects RGB but OpenCV read BGR
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img[:, :, ::-1]
        detections = self.model.detect_faces(img_rgb)

        if detections is not None and len(detections) > 0:

            for current_detection in detections:
                x, y, w, h = current_detection["box"]
                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                img_region = FacialAreaRegion(x=x, y=y, w=w, h=h)
                confidence = current_detection["confidence"]

                if align:
                    keypoints = current_detection["keypoints"]
                    left_eye = keypoints["left_eye"]
                    right_eye = keypoints["right_eye"]
                    detected_face = detection.align_face(
                        img=detected_face, left_eye=left_eye, right_eye=right_eye
                    )

                detected_face_obj = DetectedFace(
                    img=detected_face, facial_area=img_region, confidence=confidence
                )

                resp.append(detected_face_obj)

        return resp
