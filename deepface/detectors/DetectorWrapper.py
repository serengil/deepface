from typing import Any, List
import numpy as np
from deepface.models.Detector import Detector, DetectedFace
from deepface.detectors import (
    FastMtCnn,
    MediaPipe,
    MtCnn,
    OpenCv,
    Dlib,
    RetinaFace,
    Ssd,
    Yolo,
    YuNet,
)
from deepface.commons.logger import Logger

logger = Logger(module="deepface/detectors/DetectorWrapper.py")


def build_model(detector_backend: str) -> Any:
    """
    Build a face detector model
    Args:
        detector_backend (str): backend detector name
    Returns:
        built detector (Any)
    """
    global face_detector_obj  # singleton design pattern

    backends = {
        "opencv": OpenCv.OpenCvClient,
        "mtcnn": MtCnn.MtCnnClient,
        "ssd": Ssd.SsdClient,
        "dlib": Dlib.DlibClient,
        "retinaface": RetinaFace.RetinaFaceClient,
        "mediapipe": MediaPipe.MediaPipeClient,
        "yolov8": Yolo.YoloClient,
        "yunet": YuNet.YuNetClient,
        "fastmtcnn": FastMtCnn.FastMtCnnClient,
    }

    if not "face_detector_obj" in globals():
        face_detector_obj = {}

    built_models = list(face_detector_obj.keys())
    if detector_backend not in built_models:
        face_detector = backends.get(detector_backend)

        if face_detector:
            face_detector = face_detector()
            face_detector_obj[detector_backend] = face_detector
        else:
            raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face_detector_obj[detector_backend]


def detect_faces(
    detector_backend: str, img: np.ndarray, align: bool = True, expand_percentage: int = 0
) -> List[DetectedFace]:
    """
    Detect face(s) from a given image
    Args:
        detector_backend (str): detector name

        img (np.ndarray): pre-loaded image

        align (bool): enable or disable alignment after detection

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

    Returns:
        results (List[DetectedFace]): A list of DetectedFace objects
            where each object contains:

        - img (np.ndarray): The detected face as a NumPy array.

        - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h

        - confidence (float): The confidence score associated with the detected face.
    """
    face_detector: Detector = build_model(detector_backend)
    if expand_percentage < 0:
        logger.warn(
            f"Expand percentage cannot be negative but you set it to {expand_percentage}."
            "Overwritten it to 0."
        )
        expand_percentage = 0
    return face_detector.detect_faces(img=img, align=align, expand_percentage=expand_percentage)
