# built-in dependencies
from typing import Any, Dict, List, Tuple, Union

# 3rd part dependencies
import numpy as np
import cv2
from PIL import Image

# project dependencies
from deepface.modules import preprocessing
from deepface.models.Detector import DetectedFace, FacialAreaRegion
from deepface.detectors import DetectorWrapper
from deepface.commons import logger as log

logger = log.get_singletonish_logger()

# pylint: disable=no-else-raise


def extract_faces(
    img_path: Union[str, np.ndarray],
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract faces from a given image

    Args:
        img_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv)

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage

        grayscale (boolean): Flag to convert the image to grayscale before
            processing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (np.ndarray): The detected face as a NumPy array in RGB format.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values.
                left eye and right eye are eyes on the left and right respectively with respect
                to the person itself instead of observer.

        - "confidence" (float): The confidence score associated with the detected face.
    """

    resp_objs = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img, img_name = preprocessing.load_image(img_path)

    if img is None:
        raise ValueError(f"Exception while loading {img_name}")

    base_region = FacialAreaRegion(x=0, y=0, w=img.shape[1], h=img.shape[0], confidence=0)

    if detector_backend == "skip":
        face_objs = [DetectedFace(img=img, facial_area=base_region, confidence=0)]
    else:
        face_objs = DetectorWrapper.detect_faces(
            detector_backend=detector_backend,
            img=img,
            align=align,
            expand_percentage=expand_percentage,
        )

    # in case of no face found
    if len(face_objs) == 0 and enforce_detection is True:
        if img_name is not None:
            raise ValueError(
                f"Face could not be detected in {img_name}."
                "Please confirm that the picture is a face photo "
                "or consider to set enforce_detection param to False."
            )
        else:
            raise ValueError(
                "Face could not be detected. Please confirm that the picture is a face photo "
                "or consider to set enforce_detection param to False."
            )

    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [DetectedFace(img=img, facial_area=base_region, confidence=0)]

    for face_obj in face_objs:
        current_img = face_obj.img
        current_region = face_obj.facial_area

        if current_img.shape[0] == 0 or current_img.shape[1] == 0:
            continue

        if grayscale is True:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        current_img = current_img / 255  # normalize input in [0, 1]

        resp_objs.append(
            {
                "face": current_img[:, :, ::-1],
                "facial_area": {
                    "x": int(current_region.x),
                    "y": int(current_region.y),
                    "w": int(current_region.w),
                    "h": int(current_region.h),
                    "left_eye": current_region.left_eye,
                    "right_eye": current_region.right_eye,
                },
                "confidence": round(current_region.confidence, 2),
            }
        )

    if len(resp_objs) == 0 and enforce_detection == True:
        raise ValueError(
            f"Exception while extracting faces from {img_name}."
            "Consider to set enforce_detection arg to False."
        )

    return resp_objs


def align_face(
    img: np.ndarray,
    left_eye: Union[list, tuple],
    right_eye: Union[list, tuple],
) -> Tuple[np.ndarray, float]:
    """
    Align a given image horizantally with respect to their left and right eye locations
    Args:
        img (np.ndarray): pre-loaded image with detected face
        left_eye (list or tuple): coordinates of left eye with respect to the person itself
        right_eye(list or tuple): coordinates of right eye with respect to the person itself
    Returns:
        img (np.ndarray): aligned facial image
    """
    # if eye could not be detected for the given image, return image itself
    if left_eye is None or right_eye is None:
        return img, 0

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0

    angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
    img = np.array(Image.fromarray(img).rotate(angle))
    return img, angle
