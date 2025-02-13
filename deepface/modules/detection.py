# built-in dependencies
from typing import Any, Dict, IO, List, Tuple, Union, Optional, Sequence

# 3rd part dependencies
from heapq import nlargest
import numpy as np
import cv2

# project dependencies
from deepface.modules import modeling
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.commons import image_utils

from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=no-else-raise


def extract_faces(
    img_path: Union[Sequence[Union[str, np.ndarray, IO[bytes]]], str, np.ndarray, IO[bytes]],
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False,
    color_face: str = "rgb",
    normalize_face: bool = True,
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract faces from a given image or list of images

    Args:
        img_paths (List[str or np.ndarray or IO[bytes]] or str or np.ndarray or IO[bytes]): Path(s) to the image(s). Accepts exact image path
            as a string, numpy array (BGR), a file object that supports at least `.read` and is
            opened in binary mode, or base64 encoded images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv)

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage.

        grayscale (boolean): (Deprecated) Flag to convert the output face image to grayscale
            (default is False).

        color_face (string): Color to return face image output. Options: 'rgb', 'bgr' or 'gray'
            (default is 'rgb').

        normalize_face (boolean): Flag to enable normalization (divide by 255) of the output
            face image output face image normalization (default is True).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (np.ndarray): The detected face as a NumPy array in RGB format.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values.
                left eye and right eye are eyes on the left and right respectively with respect
                to the person itself instead of observer.

        - "confidence" (float): The confidence score associated with the detected face.

        - "is_real" (boolean): antispoofing analyze result. this key is just available in the
            result only if anti_spoofing is set to True in input arguments.

        - "antispoof_score" (float): score of antispoofing analyze result. this key is
            just available in the result only if anti_spoofing is set to True in input arguments.
    """

    if not isinstance(img_path, list):
        img_path = [img_path]

    all_images = []
    img_names = []

    for single_img_path in img_path:
        # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
        img, img_name = image_utils.load_image(single_img_path)

        if img is None:
            raise ValueError(f"Exception while loading {img_name}")

        all_images.append(img)
        img_names.append(img_name)

    # Run detect_faces for all images at once
    all_face_objs = detect_faces(
        detector_backend=detector_backend,
        img=all_images,
        align=align,
        expand_percentage=expand_percentage,
        max_faces=max_faces,
    )

    if len(all_images) == 1:
        all_face_objs = [all_face_objs]

    all_resp_objs = []

    for img, img_name, face_objs in zip(all_images, img_names, all_face_objs):
        height, width, _ = img.shape

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
            base_region = FacialAreaRegion(x=0, y=0, w=width, h=height, confidence=0)
            face_objs = [DetectedFace(img=img, facial_area=base_region, confidence=0)]

        for face_obj in face_objs:
            current_img = face_obj.img
            current_region = face_obj.facial_area

            if current_img.shape[0] == 0 or current_img.shape[1] == 0:
                continue

            if grayscale is True:
                logger.warn("Parameter grayscale is deprecated. Use color_face instead.")
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            else:
                if color_face == "rgb":
                    current_img = current_img[:, :, ::-1]
                elif color_face == "bgr":
                    pass  # image is in BGR
                elif color_face == "gray":
                    current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"The color_face can be rgb, bgr or gray, but it is {color_face}.")

            if normalize_face:
                current_img = current_img / 255  # normalize input in [0, 1]

            # cast to int for flask, and do final checks for borders
            x = max(0, int(current_region.x))
            y = max(0, int(current_region.y))
            w = min(width - x - 1, int(current_region.w))
            h = min(height - y - 1, int(current_region.h))

            facial_area = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "left_eye": current_region.left_eye,
                "right_eye": current_region.right_eye,
            }

            # optional nose, mouth_left and mouth_right fields are coming just for retinaface
            if current_region.nose is not None:
                facial_area["nose"] = current_region.nose
            if current_region.mouth_left is not None:
                facial_area["mouth_left"] = current_region.mouth_left
            if current_region.mouth_right is not None:
                facial_area["mouth_right"] = current_region.mouth_right

            resp_obj = {
                "face": current_img,
                "facial_area": facial_area,
                "confidence": round(float(current_region.confidence or 0), 2),
            }

            if anti_spoofing is True:
                antispoof_model = modeling.build_model(task="spoofing", model_name="Fasnet")
                is_real, antispoof_score = antispoof_model.analyze(img=img, facial_area=(x, y, w, h))
                resp_obj["is_real"] = is_real
                resp_obj["antispoof_score"] = antispoof_score

            all_resp_objs.append(resp_obj)

    return all_resp_objs


def detect_faces(
    detector_backend: str,
    img: Union[np.ndarray, List[np.ndarray]],
    align: bool = True,
    expand_percentage: int = 0,
    max_faces: Optional[int] = None,
) -> Union[List[List[DetectedFace]], List[DetectedFace]]:
    """
    Detect face(s) from a given image or list of images
    Args:
        detector_backend (str): detector name

        img (np.ndarray or List[np.ndarray]): pre-loaded image or list of images

        align (bool): enable or disable alignment after detection

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

    Returns:
        results (Union[List[List[DetectedFace]], List[DetectedFace]]): 
        A list of lists of DetectedFace objects or a list of DetectedFace objects
            where each object contains:

        - img (np.ndarray): The detected face as a NumPy array.

        - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h,
            left_eye and right eye. left eye and right eye are eyes on the left and right
            with respect to the person instead of observer.

        - confidence (float): The confidence score associated with the detected face.
    """
    if not isinstance(img, list):
        img = [img]

    face_detector: Detector = modeling.build_model(
        task="face_detector", model_name=detector_backend
    )

    preprocessed_images = []
    width_borders = []
    height_borders = []
    for single_img in img:
        height, width, _ = single_img.shape

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
            single_img = cv2.copyMakeBorder(
                single_img,
                height_border,
                height_border,
                width_border,
                width_border,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],  # Color of the border (black)
            )

        preprocessed_images.append(single_img)
        width_borders.append(width_border)
        height_borders.append(height_border)

    # Detect faces in all preprocessed images
    all_facial_areas = face_detector.detect_faces(preprocessed_images)

    if len(preprocessed_images) == 1:
        all_facial_areas = [all_facial_areas]

    all_detected_faces = []
    for single_img, facial_areas, width_border, height_border in zip(preprocessed_images, all_facial_areas, width_borders, height_borders):
        if not isinstance(facial_areas, list):
            facial_areas = [facial_areas]

        if max_faces is not None and max_faces < len(facial_areas):
            facial_areas = nlargest(
                max_faces, facial_areas, key=lambda facial_area: facial_area.w * facial_area.h
            )

        detected_faces = [
            extract_face(
                facial_area=facial_area,
                img=single_img,
                align=align,
                expand_percentage=expand_percentage,
                width_border=width_border,
                height_border=height_border,
            )
            for facial_area in facial_areas if isinstance(facial_area, FacialAreaRegion)
        ]

        all_detected_faces.append(detected_faces)

    if len(all_detected_faces) == 1:
        return all_detected_faces[0]
    return all_detected_faces


def extract_face(
    facial_area: FacialAreaRegion,
    img: np.ndarray,
    align: bool,
    expand_percentage: int,
    width_border: int,
    height_border: int,
) -> DetectedFace:
    x = facial_area.x
    y = facial_area.y
    w = facial_area.w
    h = facial_area.h
    left_eye = facial_area.left_eye
    right_eye = facial_area.right_eye
    confidence = facial_area.confidence
    nose = facial_area.nose
    mouth_left = facial_area.mouth_left
    mouth_right = facial_area.mouth_right

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
        # we were aligning the original image before, but this comes with an extra cost
        # instead we now focus on the facial area with a margin
        # and align it instead of original image to decrese the cost
        sub_img, relative_x, relative_y = extract_sub_image(img=img, facial_area=(x, y, w, h))

        aligned_sub_img, angle = align_img_wrt_eyes(
            img=sub_img, left_eye=left_eye, right_eye=right_eye
        )

        rotated_x1, rotated_y1, rotated_x2, rotated_y2 = project_facial_area(
            facial_area=(
                relative_x,
                relative_y,
                relative_x + w,
                relative_y + h,
            ),
            angle=angle,
            size=(sub_img.shape[0], sub_img.shape[1]),
        )
        detected_face = aligned_sub_img[
            int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
        ]

        # do not spend memory for these temporary variables anymore
        del aligned_sub_img, sub_img

        # restore x, y, le and re before border added
        x = x - width_border
        y = y - height_border
        # w and h will not change
        if left_eye is not None:
            left_eye = (left_eye[0] - width_border, left_eye[1] - height_border)
        if right_eye is not None:
            right_eye = (right_eye[0] - width_border, right_eye[1] - height_border)
        if nose is not None:
            nose = (nose[0] - width_border, nose[1] - height_border)
        if mouth_left is not None:
            mouth_left = (mouth_left[0] - width_border, mouth_left[1] - height_border)
        if mouth_right is not None:
            mouth_right = (mouth_right[0] - width_border, mouth_right[1] - height_border)

    return DetectedFace(
        img=detected_face,
        facial_area=FacialAreaRegion(
            x=x,
            y=y,
            h=h,
            w=w,
            confidence=confidence,
            left_eye=left_eye,
            right_eye=right_eye,
            nose=nose,
            mouth_left=mouth_left,
            mouth_right=mouth_right,
        ),
        confidence=confidence or 0,
    )


def extract_sub_image(
    img: np.ndarray, facial_area: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, int, int]:
    """
    Get the sub image with given facial area while expanding the facial region
        to ensure alignment does not shift the face outside the image.

    This function doubles the height and width of the face region,
    and adds black pixels if necessary.

    Args:
        - img (np.ndarray): pre-loaded image with detected face
        - facial_area (tuple of int): Representing the (x, y, w, h) of the facial area.

    Returns:
        - extracted_face (np.ndarray): expanded facial image
        - relative_x (int): adjusted x-coordinates relative to the expanded region
        - relative_y (int): adjusted y-coordinates relative to the expanded region
    """
    x, y, w, h = facial_area
    relative_x = int(0.5 * w)
    relative_y = int(0.5 * h)

    # calculate expanded coordinates
    x1, y1 = x - relative_x, y - relative_y
    x2, y2 = x + w + relative_x, y + h + relative_y

    # most of the time, the expanded region fits inside the image
    if x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
        return img[y1:y2, x1:x2], relative_x, relative_y

    # but sometimes, we need to add black pixels
    # ensure the coordinates are within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    cropped_region = img[y1:y2, x1:x2]

    # create a black image
    extracted_face = np.zeros(
        (h + 2 * relative_y, w + 2 * relative_x, img.shape[2]), dtype=img.dtype
    )

    # map the cropped region
    start_x = max(0, relative_x - x)
    start_y = max(0, relative_y - y)
    extracted_face[
        start_y : start_y + cropped_region.shape[0], start_x : start_x + cropped_region.shape[1]
    ] = cropped_region

    return extracted_face, relative_x, relative_y


def align_img_wrt_eyes(
    img: np.ndarray,
    left_eye: Optional[Union[list, tuple]],
    right_eye: Optional[Union[list, tuple]],
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

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    return img, angle


def project_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Update pre-calculated facial area coordinates after image itself
        rotated with respect to the eyes.
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
