from typing import List, Tuple
import numpy as np
from retinaface import RetinaFace as rf
from retinaface.commons import postprocess
from deepface.models.Detector import Detector

# pylint: disable=too-few-public-methods
class RetinaFaceClient(Detector):
    def __init__(self):
        self.model = rf.build_model()

    def detect_faces(
        self, img: np.ndarray, align: bool = True
    ) -> List[Tuple[np.ndarray, List[float], float]]:
        """
        Detect and align face with retinaface
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

        obj = rf.detect_faces(img, model=self.model, threshold=0.9)

        if isinstance(obj, dict):
            for face_idx in obj.keys():
                identity = obj[face_idx]
                facial_area = identity["facial_area"]

                y = facial_area[1]
                h = facial_area[3] - y
                x = facial_area[0]
                w = facial_area[2] - x
                img_region = [x, y, w, h]
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

                resp.append((detected_face, img_region, confidence))

        return resp
