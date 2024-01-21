# built-in dependencies
from typing import Any, Dict, List, Tuple, Union

# 3rd part dependencies
import numpy as np
from PIL import Image

# project dependencies
from deepface.commons import functions


def extract_faces(
    img_path: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    grayscale: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract faces from a given image

    Args:
        img_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv)

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        align (bool): Flag to enable face alignment (default is True).

        grayscale (boolean): Flag to convert the image to grayscale before
            processing (default is False).


    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:
        - "face" (np.ndarray): The detected face as a NumPy array.
        - "facial_area" (List[float]): The detected face's regions represented as a list of floats.
        - "confidence" (float): The confidence score associated with the detected face.
    """

    resp_objs = []

    img_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=grayscale,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img, region, confidence in img_objs:
        resp_obj = {}

        # discard expanded dimension
        if len(img.shape) == 4:
            img = img[0]

        # bgr to rgb
        resp_obj["face"] = img[:, :, ::-1]
        resp_obj["facial_area"] = region
        resp_obj["confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs


def align_face(
    img: np.ndarray,
    left_eye: Union[list, tuple],
    right_eye: Union[list, tuple],
) -> np.ndarray:
    """
    Align a given image horizantally with respect to their left and right eye locations
    Args:
        img (np.ndarray): pre-loaded image with detected face
        left_eye (list or tuple): coordinates of left eye with respect to the you
        right_eye(list or tuple): coordinates of right eye with respect to the you
    Returns:
        img (np.ndarray): aligned facial image
    """
    # if eye could not be detected for the given image, return image itself
    if left_eye is None or right_eye is None:
        return img

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    img = Image.fromarray(img)
    img = np.array(img.rotate(angle))
    return img
