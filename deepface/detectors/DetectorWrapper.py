from typing import List, Tuple
import numpy as np
import cv2
from deepface.modules import detection, modeling
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()


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

        - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h,
            left_eye and right eye. left eye and right eye are eyes on the left and right
            with respect to the person instead of observer.

        - confidence (float): The confidence score associated with the detected face.
    """
    height, width, _ = img.shape

    face_detector: Detector = modeling.build_model(
        task="face_detector", model_name=detector_backend
    )

    # validate expand percentage score
    if expand_percentage < 0:
        logger.warn(
            f"Expand percentage cannot be negative but you set it to {expand_percentage}."
            "Overwritten it to 0."
        )
        expand_percentage = 0

    # If faces are close to the upper boundary, alignment move them outside
    # Add a black border around an image to avoid this.
    height_border = int(0.5 * height)
    width_border = int(0.5 * width)
    if align is True:
        img = cv2.copyMakeBorder(
            img,
            height_border,
            height_border,
            width_border,
            width_border,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],  # Color of the border (black)
        )

    # find facial areas of given image
    facial_areas = face_detector.detect_faces(img)

    results = []
    for facial_area in facial_areas:
        x = facial_area.x
        y = facial_area.y
        w = facial_area.w
        h = facial_area.h
        left_eye = facial_area.left_eye
        right_eye = facial_area.right_eye
        confidence = facial_area.confidence

        if expand_percentage > 0:
            # Expand the facial region height and width by the provided percentage
            # ensuring that the expanded region stays within img.shape limits
            expanded_w = w + int(w * expand_percentage / 100)
            expanded_h = h + int(h * expand_percentage / 100)

            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(img.shape[1] - x, expanded_w)
            h = min(img.shape[0] - y, expanded_h)

        # extract detected face unaligned
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

        # align original image, then find projection of detected face area after alignment
        if align is True:  # and left_eye is not None and right_eye is not None:
            aligned_img, angle = detection.align_face(
                img=img, left_eye=left_eye, right_eye=right_eye
            )

            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
                facial_area=(x, y, x + w, y + h), angle=angle, size=(img.shape[0], img.shape[1])
            )
            detected_face = aligned_img[
                int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
            ]

            # restore x, y, le and re before border added
            x = x - width_border
            y = y - height_border
            # w and h will not change
            if left_eye is not None:
                left_eye = (left_eye[0] - width_border, left_eye[1] - height_border)
            if right_eye is not None:
                right_eye = (right_eye[0] - width_border, right_eye[1] - height_border)

        result = DetectedFace(
            img=detected_face,
            facial_area=FacialAreaRegion(
                x=x, y=y, h=h, w=w, confidence=confidence, left_eye=left_eye, right_eye=right_eye
            ),
            confidence=confidence,
        )
        results.append(result)
    return results


def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Rotate the facial area around its center.
    Inspried from the work of @UmutDeniz26 - github.com/serengil/retinaface/pull/80

    Args:
        facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
            x2 is equal to x1 + w1, and y2 is equal to y1 + h1
        angle (float): Angle of rotation in degrees. Its sign determines the direction of rotation.
                       Note that angles > 360 degrees are normalized to the range [0, 360).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        rotated_coordinates (tuple of int): Representing the new coordinates
            (x1, y1, x2, y2) or (x1, y1, x1+w1, y1+h1) of the rotated facial area.
    """

    # Normalize the witdh of the angle so we don't have to
    # worry about rotations greater than 360 degrees.
    # We workaround the quirky behavior of the modulo operator
    # for negative angle values.
    direction = 1 if angle >= 0 else -1
    angle = abs(angle) % 360
    if angle == 0:
        return facial_area

    # Angle in radians
    angle = angle * np.pi / 180

    height, weight = size

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + weight / 2
    y_new = y_new + height / 2

    # Calculate projected coordinates after alignment
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    # validate projected coordinates are in image's boundaries
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)

    return (x1, y1, x2, y2)
