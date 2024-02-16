from typing import Any, List
import numpy as np
from deepface.modules import detection
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
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

    # validate expand percentage score
    if expand_percentage < 0:
        logger.warn(
            f"Expand percentage cannot be negative but you set it to {expand_percentage}."
            "Overwritten it to 0."
        )
        expand_percentage = 0

    # find facial areas of given image
    facial_areas = face_detector.detect_faces(img=img)

    results = []
    for facial_area in facial_areas:
        x = facial_area.x
        y = facial_area.y
        w = facial_area.w
        h = facial_area.h
        left_eye = facial_area.left_eye
        right_eye = facial_area.right_eye
        confidence = facial_area.confidence

        # expand the facial area to be extracted and stay within img.shape limits
        x2 = max(0, x - int((w * expand_percentage) / 100))  # expand left
        y2 = max(0, y - int((h * expand_percentage) / 100))  # expand top
        w2 = min(img.shape[1], w + int((w * expand_percentage) / 100))  # expand right
        h2 = min(img.shape[0], h + int((h * expand_percentage) / 100))  # expand bottom

        # extract detected face unaligned
        detected_face = img[int(y2) : int(y2 + h2), int(x2) : int(x2 + w2)]

        # align detected face
        if align is True:
            detected_face = detection.align_face(
                img=detected_face, left_eye=left_eye, right_eye=right_eye
            )

        result = DetectedFace(
            img=detected_face,
            facial_area=FacialAreaRegion(
                x=x, y=y, h=h, w=w, confidence=confidence, left_eye=left_eye, right_eye=right_eye
            ),
            confidence=confidence,
        )
        results.append(result)
    return results
