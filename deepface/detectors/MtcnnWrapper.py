from typing import Any
import cv2
import numpy as np
from deepface.detectors import FaceDetector


def build_model() -> Any:
    """
    Build a mtcnn face detector model
    Returns:
        model (Any)
    """
    from mtcnn import MTCNN

    face_detector = MTCNN()
    return face_detector


def detect_face(face_detector: Any, img: np.ndarray, align: bool = True) -> list:
    """
    Detect and align face with mtcnn
    Args:
        face_detector (mtcnn.MTCNN): mtcnn face detector object
        img (np.ndarray): pre-loaded image
        align (bool): default is true
    Returns:
        list of detected and aligned faces
    """

    resp = []

    detected_face = None
    img_region = [0, 0, img.shape[1], img.shape[0]]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
    detections = face_detector.detect_faces(img_rgb)

    if len(detections) > 0:

        for detection in detections:
            x, y, w, h = detection["box"]
            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
            img_region = [x, y, w, h]
            confidence = detection["confidence"]

            if align:
                keypoints = detection["keypoints"]
                left_eye = keypoints["left_eye"]
                right_eye = keypoints["right_eye"]
                detected_face = FaceDetector.alignment_procedure(detected_face, left_eye, right_eye)

            resp.append((detected_face, img_region, confidence))

    return resp
