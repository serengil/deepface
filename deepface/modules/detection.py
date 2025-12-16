# built-in dependencies
from typing import Any, Dict, IO, List, Tuple, Union, Optional

# 3rd part dependencies
from heapq import nlargest
import numpy as np
import cv2

# project dependencies
from deepface.modules import modeling
from deepface.models.Face import Face, FaceSpoofing, FaceLandmarks
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.commons import image_utils

from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=no-else-raise

def translate_tuple(t: Optional[Tuple[int, int]], offset: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if t is None:
        return None
    return (t[0] + offset[0], t[1] + offset[1])

def extract_faces(
    img_path: Union[str, np.ndarray, IO[bytes]],
    detector_backend: str = "opencv",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> Tuple[np.ndarray, List[Face]]:
    """
    Extract faces from a given image.

    Args:
        img_path (str or np.ndarray or IO[bytes]): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), a file object that supports at least `.read` and is
            opened in binary mode, or base64 encoded images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' (default is opencv)

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage.

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed and returned (default is None).

    Returns:
        Tuple[np.ndarray, List[Face]]: A tuple containing the original image as a numpy array and a list of Face objects,
            each containing a valid extracted face and related metadata alongside spoofing analysis if enabled.
    """

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img, img_name = image_utils.load_image(img_path)
    if img is None:
        raise ValueError(f"Exception while loading {img_name}")

    # Add black borders to the image to allow face detection near edges
    # Note: this step is critical for face detection but it is very important NOT to use 
    # those black borders for other analysis later on (especially spoofing).
    height, width, _ = img.shape
    height_border = int(0.5 * height)
    width_border = int(0.5 * width)
    border_img = cv2.copyMakeBorder(
        img,
        height_border,
        height_border,
        width_border,
        width_border,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],  # Color of the border (black)
    )

    # Detect up to max_faces faces in the image
    face_detector: Detector = modeling.build_model(task="face_detector", model_name=detector_backend)
    facial_areas = face_detector.detect_faces(border_img)
    del border_img # free memory and ensure we do not use it later by mistake
    if max_faces is not None and max_faces < len(facial_areas):
        facial_areas = nlargest(
            max_faces,
            facial_areas,
            key=lambda facial_area: facial_area.w * facial_area.h
        )


    # Build Face objects for each detected valid facial area
    # and re-calibrated facial area to original image coordinates
    faces: List[Face] = []
    for fa in facial_areas:
        face = Face(
            img=img,
            landmarks=FaceLandmarks(
                x=fa.x - width_border,
                y=fa.y - height_border,
                w=fa.w,
                h=fa.h,
                left_eye=translate_tuple(fa.left_eye, (-width_border, -height_border)),
                right_eye=translate_tuple(fa.right_eye, (-width_border, -height_border)),
                nose=translate_tuple(fa.nose, (-width_border, -height_border)),
                mouth_left=translate_tuple(fa.mouth_left, (-width_border, -height_border)),
                mouth_right=translate_tuple(fa.mouth_right, (-width_border, -height_border)),
            ),
            confidence=fa.confidence,
        )
        if face.landmarks.is_valid(width, height):
            faces.append(face)

    # Perform anti-spoofing analysis if enabled for each detected face
    if anti_spoofing is True:
        antispoof_model = modeling.build_model(task="spoofing", model_name="Fasnet")
        for face in faces:
            spoof_confidence, real_confidence, uncertainty = antispoof_model.analyze(
                img=img, 
                facial_area=(face.landmarks.x, face.landmarks.y, face.landmarks.w, face.landmarks.h)
            )
            face.spoofing = FaceSpoofing(
                spoof_confidence=spoof_confidence,
                real_confidence=real_confidence,
                uncertainty_confidence=uncertainty,
            )

    return img, faces

def is_valid_landmark(coord: Optional[Union[tuple, list]], width: int, height: int) -> bool:
    """
    Check if a landmark coordinate is within valid image bounds.

    Args:
        coord (tuple or list or None): (x, y) coordinate to check.
        width (int): Image width.
        height (int): Image height.
    Returns:
        bool: True if coordinate is valid and within bounds, False otherwise.
    """
    if coord is None:
        return False
    if not (isinstance(coord, (tuple, list)) and len(coord) == 2):
        return False
    x, y = coord
    return 0 <= x < width and 0 <= y < height
