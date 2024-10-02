# built-in dependencies
from typing import Any, List

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion


class MediaPipeClient(Detector):
    """
    MediaPipe from google.github.io/mediapipe/solutions/face_detection
    """

    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a mediapipe face detector model
        Returns:
            model (Any)
        """
        # this is not a must dependency. do not import it in the global level.
        try:
            import mediapipe as mp
        except ModuleNotFoundError as e:
            raise ImportError(
                "MediaPipe is an optional detector, ensure the library is installed. "
                "Please install using 'pip install mediapipe'"
            ) from e

        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
        return face_detection

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with mediapipe

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        img_width = img.shape[1]
        img_height = img.shape[0]

        results = self.model.process(img)

        # If no face has been detected, return an empty list
        if results.detections is None:
            return resp

        # Extract the bounding box, the landmarks and the confidence score
        for current_detection in results.detections:
            (confidence,) = current_detection.score

            bounding_box = current_detection.location_data.relative_bounding_box
            landmarks = current_detection.location_data.relative_keypoints

            x = int(bounding_box.xmin * img_width)
            w = int(bounding_box.width * img_width)
            y = int(bounding_box.ymin * img_height)
            h = int(bounding_box.height * img_height)

            right_eye = (int(landmarks[0].x * img_width), int(landmarks[0].y * img_height))
            left_eye = (int(landmarks[1].x * img_width), int(landmarks[1].y * img_height))
            # nose = (int(landmarks[2].x * img_width), int(landmarks[2].y * img_height))
            # mouth = (int(landmarks[3].x * img_width), int(landmarks[3].y * img_height))
            # right_ear = (int(landmarks[4].x * img_width), int(landmarks[4].y * img_height))
            # left_ear = (int(landmarks[5].x * img_width), int(landmarks[5].y * img_height))

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=float(confidence),
            )
            resp.append(facial_area)

        return resp
