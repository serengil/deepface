from typing import Any, Union
from PIL import Image
import numpy as np
from deepface.detectors import (
    OpenCvWrapper,
    SsdWrapper,
    DlibWrapper,
    MtcnnWrapper,
    RetinaFaceWrapper,
    MediapipeWrapper,
    YoloWrapper,
    YunetWrapper,
    FastMtcnnWrapper,
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
        "opencv": OpenCvWrapper.build_model,
        "ssd": SsdWrapper.build_model,
        "dlib": DlibWrapper.build_model,
        "mtcnn": MtcnnWrapper.build_model,
        "retinaface": RetinaFaceWrapper.build_model,
        "mediapipe": MediapipeWrapper.build_model,
        "yolov8": YoloWrapper.build_model,
        "yunet": YunetWrapper.build_model,
        "fastmtcnn": FastMtcnnWrapper.build_model,
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


def detect_face(
    face_detector: Any, detector_backend: str, img: np.ndarray, align: bool = True
) -> tuple:
    """
    Detect a single face from a given image
    Args:
        face_detector (Any): pre-built face detector object
        detector_backend (str): detector name
        img (np.ndarray): pre-loaded image
        alig (bool): enable or disable alignment after detection
    Returns
        result (tuple): tuple of face (np.ndarray), face region (list)
            , confidence score (float)
    """
    obj = detect_faces(face_detector, detector_backend, img, align)

    if len(obj) > 0:
        face, region, confidence = obj[0]  # discard multiple faces

    # If no face is detected, set face to None,
    # image region to full image, and confidence to 0.
    else:  # len(obj) == 0
        face = None
        region = [0, 0, img.shape[1], img.shape[0]]
        confidence = 0

    return face, region, confidence


def detect_faces(
    face_detector: Any, detector_backend: str, img: np.ndarray, align: bool = True
) -> list:
    """
    Detect face(s) from a given image
    Args:
        face_detector (Any): pre-built face detector object
        detector_backend (str): detector name
        img (np.ndarray): pre-loaded image
        alig (bool): enable or disable alignment after detection
    Returns
        result (list): tuple of face (np.ndarray), face region (list)
            , confidence score (float)
    """
    backends = {
        "opencv": OpenCvWrapper.detect_face,
        "ssd": SsdWrapper.detect_face,
        "dlib": DlibWrapper.detect_face,
        "mtcnn": MtcnnWrapper.detect_face,
        "retinaface": RetinaFaceWrapper.detect_face,
        "mediapipe": MediapipeWrapper.detect_face,
        "yolov8": YoloWrapper.detect_face,
        "yunet": YunetWrapper.detect_face,
        "fastmtcnn": FastMtcnnWrapper.detect_face,
    }

    detect_face_fn = backends.get(detector_backend)

    if detect_face_fn:  # pylint: disable=no-else-return
        obj = detect_face_fn(face_detector, img, align)
        # obj stores list of (detected_face, region, confidence)
        return obj
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)


def get_alignment_angle_arctan2(
    left_eye: Union[list, tuple], right_eye: Union[list, tuple]
) -> float:
    """
    Find the angle between eyes
    Args:
        left_eye: coordinates of left eye with respect to the you
        right_eye: coordinates of right eye with respect to the you
    Returns:
        angle (float)
    """
    return float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))


def alignment_procedure(
    img: np.ndarray, left_eye: Union[list, tuple], right_eye: Union[list, tuple]
) -> np.ndarray:
    """
    Rotate given image until eyes are on a horizontal line
    Args:
        img (np.ndarray): pre-loaded image
        left_eye: coordinates of left eye with respect to the you
        right_eye: coordinates of right eye with respect to the you
    Returns:
        result (np.ndarray): aligned face
    """
    angle = get_alignment_angle_arctan2(left_eye, right_eye)
    img = Image.fromarray(img)
    img = np.array(img.rotate(angle))
    return img
