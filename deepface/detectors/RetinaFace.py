from typing import List
import numpy as np
from retinaface import RetinaFace as rf
from retinaface.commons import postprocess
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion

# pylint: disable=too-few-public-methods
class RetinaFaceClient(Detector):
    def __init__(self):
        self.model = rf.build_model()

    def detect_faces(
        self, img: np.ndarray, align: bool = True, expand_percentage: int = 0
    ) -> List[DetectedFace]:
        """
        Detect and align face with retinaface

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

        obj = rf.detect_faces(img, model=self.model, threshold=0.9)

        if not isinstance(obj, dict):
            return resp

        for face_idx in obj.keys():
            identity = obj[face_idx]
            facial_area = identity["facial_area"]

            y = facial_area[1]
            h = facial_area[3] - y
            x = facial_area[0]
            w = facial_area[2] - x
            img_region = FacialAreaRegion(x=x, y=y, w=w, h=h)
            confidence = identity["score"]

            # expand the facial area to be extracted and stay within img.shape limits
            x2 = max(0, x - int((w * expand_percentage) / 100))  # expand left
            y2 = max(0, y - int((h * expand_percentage) / 100))  # expand top
            w2 = min(img.shape[1], w + int((w * expand_percentage) / 100))  # expand right
            h2 = min(img.shape[0], h + int((h * expand_percentage) / 100))  # expand bottom

            # detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
            detected_face = img[int(y2) : int(y2 + h2), int(x2) : int(x2 + w2)]

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
