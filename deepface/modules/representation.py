# built-in dependencies
from typing import Any, Dict, List, Union

# 3rd party dependencies
import numpy as np
import cv2

# project dependencies
from deepface.modules import modeling
from deepface.commons import functions
from deepface.models.FacialRecognition import FacialRecognition


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    normalization: str = "base",
) -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8'.

        align (boolean): Perform alignment based on the eye positions.

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - embedding (np.array): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).
        - facial_area (dict): Detected facial area by face detection in dictionary format.
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.
        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.
    """
    resp_objs = []

    model: FacialRecognition = modeling.build_model(model_name)

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = functions.find_target_size(model_name=model_name)
    if detector_backend != "skip":
        img_objs = functions.extract_faces(
            img=img_path,
            target_size=(target_size[1], target_size[0]),
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    else:  # skip
        # Try load. If load error, will raise exception internal
        img, _ = functions.load_image(img_path)
        # --------------------------------
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
            # when called from verify, this is already normalized. But needed when user given.
            if img.max() > 1:
                img = (img.astype(np.float32) / 255.0).astype(np.float32)
        # --------------------------------
        # make dummy region and confidence to keep compatibility with `extract_faces`
        img_region = {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]}
        img_objs = [(img, img_region, 0)]
    # ---------------------------------

    for img, region, confidence in img_objs:
        # custom normalization
        img = functions.normalize_input(img=img, normalization=normalization)

        embedding = model.find_embeddings(img)

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs
