from typing import Any
import numpy as np
from deepface.models.Detector import Detector
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


def detect_faces(detector_backend: str, img: np.ndarray, align: bool = True) -> list:
    """
    Detect face(s) from a given image
    Args:
        detector_backend (str): detector name
        img (np.ndarray): pre-loaded image
        alig (bool): enable or disable alignment after detection
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
    face_detector: Detector = build_model(detector_backend)
    return face_detector.detect_faces(img=img, align=align)
