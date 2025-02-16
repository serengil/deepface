# built-in dependencies
from typing import List, Union

# 3rd party dependencies
import numpy as np
from retinaface import RetinaFace as rf

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion

# pylint: disable=too-few-public-methods
class RetinaFaceClient(Detector):
    def __init__(self):
        self.model = rf.build_model()

    def detect_faces(self, img: Union[np.ndarray, List[np.ndarray]]) -> Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]:
        """
        Detect and align faces with retinaface in an image or a list of images

        Args:
            img (Union[np.ndarray, List[np.ndarray]]): pre-loaded image as numpy array or a list of those

        Returns:
            results (Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]): A list or a list of lists of FacialAreaRegion objects
        """
        if isinstance(img, np.ndarray):
            imgs = [img]
        else:
            imgs = img

        batch_results = []

        for img in imgs:
            resp = []
            obj = rf.detect_faces(img, model=self.model, threshold=0.9)

            if isinstance(obj, dict):
                for face_idx in obj.keys():
                    identity = obj[face_idx]
                    detection = identity["facial_area"]

                    y = detection[1]
                    h = detection[3] - y
                    x = detection[0]
                    w = detection[2] - x

                    left_eye = tuple(int(i) for i in identity["landmarks"]["left_eye"])
                    right_eye = tuple(int(i) for i in identity["landmarks"]["right_eye"])
                    nose = identity["landmarks"].get("nose")
                    mouth_right = identity["landmarks"].get("mouth_right")
                    mouth_left = identity["landmarks"].get("mouth_left")

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

            batch_results.append(resp)

        return batch_results if len(batch_results) > 1 else batch_results[0]
