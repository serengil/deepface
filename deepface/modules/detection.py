# built-in dependencies
from typing import Any, Dict, List, Tuple, Union

# 3rd part dependencies
import numpy as np

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
    This function applies pre-processing stages of a face recognition pipeline
    including detection and alignment

    Parameters:
            img_path: exact image path, numpy array (BGR) or base64 encoded image.
            Source image can have many face. Then, result will be the size of number
            of faces appearing in that source image.

            target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image.

            detector_backend (string): face detection backends are retinaface, mtcnn,
            opencv, ssd or dlib

            enforce_detection (boolean): function throws exception if face cannot be
            detected in the fed image. Set this to False if you do not want to get
            an exception and run the function anyway.

            align (boolean): alignment according to the eye positions.

            grayscale (boolean): extracting faces in rgb or gray scale

    Returns:
            list of dictionaries. Each dictionary will have facial image itself (RGB),
            extracted area from the original image and confidence score.

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
