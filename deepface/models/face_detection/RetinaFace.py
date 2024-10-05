# built-in dependencies
from typing import List

# 3rd party dependencies
import numpy as np
from retinaface import RetinaFace as rf

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion

# pylint: disable=too-few-public-methods
class RetinaFaceClient(Detector):
    def __init__(self):
        self.model = rf.build_model()

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with retinaface

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        obj = rf.detect_faces(img, model=self.model, threshold=0.9)

        if not isinstance(obj, dict):
            return resp

        for face_idx in obj.keys():
            identity = obj[face_idx]
            detection = identity["facial_area"]

            y = detection[1]
            h = detection[3] - y
            x = detection[0]
            w = detection[2] - x

            # retinaface sets left and right eyes with respect to the person
            left_eye = identity["landmarks"]["left_eye"]
            right_eye = identity["landmarks"]["right_eye"]
            nose = identity["landmarks"].get("nose")
            mouth_right = identity["landmarks"].get("mouth_right")
            mouth_left = identity["landmarks"].get("mouth_left")

            # eyes are list of float, need to cast them tuple of int
            left_eye = tuple(int(i) for i in left_eye)
            right_eye = tuple(int(i) for i in right_eye)
            if nose is not None:
                nose = tuple(int(i) for i in nose)
            if mouth_right is not None:
                mouth_right = tuple(int(i) for i in mouth_right)
            if mouth_left is not None:
                mouth_left = tuple(int(i) for i in mouth_left)

            confidence = identity["score"]

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
                nose=nose,
                mouth_left=mouth_left,
                mouth_right=mouth_right,
            )

            resp.append(facial_area)

        return resp
