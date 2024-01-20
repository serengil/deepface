from typing import Any, Union
import cv2
import numpy as np
from deepface.models.Detector import Detector

# Link -> https://github.com/timesler/facenet-pytorch
# Examples https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch


class FastMtCnn(Detector):
    def __init__(self):
        self.model = self.build_model()

    def detect_faces(self, img: np.ndarray, align: bool = True) -> list:
        """
        Detect and align face with mtcnn
        Args:
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            list of detected and aligned faces
        """
        resp = []

        detected_face = None
        img_region = [0, 0, img.shape[1], img.shape[0]]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
        detections = self.model.detect(
            img_rgb, landmarks=True
        )  # returns boundingbox, prob, landmark
        if len(detections[0]) > 0:

            for detection in zip(*detections):
                x, y, w, h = xyxy_to_xywh(detection[0])
                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                img_region = [x, y, w, h]
                confidence = detection[1]

                if align:
                    left_eye = detection[2][0]
                    right_eye = detection[2][1]
                    detected_face = self.align_face(
                        img=detected_face, left_eye=left_eye, right_eye=right_eye
                    )

                resp.append((detected_face, img_region, confidence))

        return resp

    def build_model(self) -> Any:
        """
        Build a fast mtcnn face detector model
        Returns:
            model (Any)
        """
        # this is not a must dependency. do not import it in the global level.
        try:
            from facenet_pytorch import MTCNN as fast_mtcnn
        except ModuleNotFoundError as e:
            raise ImportError(
                "FastMtcnn is an optional detector, ensure the library is installed."
                "Please install using 'pip install facenet-pytorch' "
            ) from e

        face_detector = fast_mtcnn(
            image_size=160,
            thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
            post_process=True,
            device="cpu",
            select_largest=False,  # return result in descending order
        )
        return face_detector


def xyxy_to_xywh(xyxy: Union[list, tuple]) -> list:
    """
    Convert xyxy format to xywh format.
    """
    x, y = xyxy[0], xyxy[1]
    w = xyxy[2] - x + 1
    h = xyxy[3] - y + 1
    return [x, y, w, h]
