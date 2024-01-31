import os
from typing import Any, List
import numpy as np
import gdown
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.modules import detection
from deepface.commons.logger import Logger

logger = Logger()

# Model's weights paths
PATH = "/.deepface/weights/yolov8n-face.pt"

# Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
WEIGHT_URL = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"

# Confidence thresholds for landmarks detection
# used in alignment_procedure function
LANDMARKS_CONFIDENCE_THRESHOLD = 0.5


class YoloClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """

        # Import the Ultralytics YOLO model
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector, ensure the library is installed. \
                Please install using 'pip install ultralytics' "
            ) from e

        from deepface.commons.functions import get_deepface_home

        weight_path = f"{get_deepface_home()}{PATH}"

        # Download the model's weights if they don't exist
        if not os.path.isfile(weight_path):
            gdown.download(WEIGHT_URL, weight_path, quiet=False)
            logger.info(f"Downloaded YOLO model {os.path.basename(weight_path)}")

        # Return face_detector
        return YOLO(weight_path)

    def detect_faces(
        self, img: np.ndarray, align: bool = False, expand_percentage: int = 0
    ) -> List[DetectedFace]:
        """
        Detect and align face with yolo

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

        # Detect faces
        results = self.model.predict(img, verbose=False, show=False, conf=0.25)[0]

        # For each face, extract the bounding box, the landmarks and confidence
        for result in results:
            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            region = FacialAreaRegion(x=x, y=y, w=w, h=h)

            # expand the facial area to be extracted and stay within img.shape limits
            x2 = max(0, x - int((w * expand_percentage) / 100))  # expand left
            y2 = max(0, y - int((h * expand_percentage) / 100))  # expand top
            w2 = min(img.shape[1], w + int((w * expand_percentage) / 100))  # expand right
            h2 = min(img.shape[0], h + int((h * expand_percentage) / 100))  # expand bottom

            # detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
            detected_face = img[int(y2) : int(y2 + h2), int(x2) : int(x2 + w2)]

            if align:
                # Tuple of x,y and confidence for left eye
                left_eye = result.keypoints.xy[0][0], result.keypoints.conf[0][0]
                # Tuple of x,y and confidence for right eye
                right_eye = result.keypoints.xy[0][1], result.keypoints.conf[0][1]

                # Check the landmarks confidence before alignment
                if (
                    left_eye[1] > LANDMARKS_CONFIDENCE_THRESHOLD
                    and right_eye[1] > LANDMARKS_CONFIDENCE_THRESHOLD
                ):
                    detected_face = detection.align_face(
                        img=detected_face, left_eye=left_eye[0].cpu(), right_eye=right_eye[0].cpu()
                    )

            detected_face_obj = DetectedFace(
                img=detected_face, facial_area=region, confidence=confidence
            )
            resp.append(detected_face_obj)

        return resp
