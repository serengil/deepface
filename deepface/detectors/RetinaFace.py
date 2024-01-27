from typing import List
import numpy as np
from retinaface import RetinaFace as rf
from retinaface.commons import postprocess
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion

# pylint: disable=too-few-public-methods
class RetinaFaceClient(Detector):
    def __init__(self):
        self.model = rf.build_model()

    def detect_faces(self, img: np.ndarray, align: bool = True) -> List[DetectedFace]:
        """
        Detect and align face with retinaface
        Args:
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            results (List[DetectedFace]): A list of DetectedFace object
                where each object contains:
            - img (np.ndarray): The detected face as a NumPy array.
            - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h
            - confidence (float): The confidence score associated with the detected face.
        """
        resp = []

        obj = rf.detect_faces(img, model=self.model, threshold=0.9)

        if isinstance(obj, dict):
            for face_idx in obj.keys():
                identity = obj[face_idx]
                facial_area = identity["facial_area"]

                y = facial_area[1]
                h = facial_area[3] - y
                x = facial_area[0]
                w = facial_area[2] - x
                img_region = FacialAreaRegion(x=x, y=y, w=w, h=h)
                confidence = identity["score"]

                # detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
                detected_face = img[
                    facial_area[1] : facial_area[3], facial_area[0] : facial_area[2]
                ]

                if align:
                    landmarks = identity["landmarks"]
                    left_eye = landmarks["left_eye"]
                    right_eye = landmarks["right_eye"]
                    nose = landmarks["nose"]
                    # mouth_right = landmarks["mouth_right"]
                    # mouth_left = landmarks["mouth_left"]

                    detected_face = postprocess.alignment_procedure(
                        detected_face, right_eye, left_eye, nose
                    )

                detected_face_obj = DetectedFace(
                    img=detected_face,
                    facial_area=img_region,
                    confidence=confidence,
                )

                resp.append(detected_face_obj)

        return resp
