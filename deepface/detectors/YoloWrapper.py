from typing import Any
import numpy as np
from deepface.models.Detector import Detector
from deepface.commons.logger import Logger

logger = Logger()

# Model's weights paths
PATH = "/.deepface/weights/yolov8n-face.pt"

# Google Drive URL
WEIGHT_URL = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"

# Confidence thresholds for landmarks detection
# used in alignment_procedure function
LANDMARKS_CONFIDENCE_THRESHOLD = 0.5


class Yolo(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """
        import gdown
        import os

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

    def detect_faces(self, img: np.ndarray, align: bool = False) -> list:
        """
        Detect and align face with yolo
        Args:
            face_detector (Any): yolo face detector object
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            list of detected and aligned faces
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
            detected_face = img[y : y + h, x : x + w].copy()

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
                    detected_face = self.align_face(
                        img=detected_face, left_eye=left_eye[0].cpu(), right_eye=right_eye[0].cpu()
                    )
            resp.append((detected_face, [x, y, w, h], confidence))

        return resp
