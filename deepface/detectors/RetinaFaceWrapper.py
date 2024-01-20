import numpy as np
from retinaface import RetinaFace as rf
from retinaface.commons import postprocess
from deepface.models.Detector import Detector


class RetinaFace(Detector):
    def __init__(self):
        self.model = rf.build_model()

    def detect_faces(self, img: np.ndarray, align: bool = True) -> list:
        """
        Detect and align face with retinaface
        Args:
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            list of detected and aligned faces
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
