# built-in dependencies
import time
from typing import Any, Dict, Optional, Union, List, Tuple

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules import representation, detection, modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons import logger as log

logger = log.get_singletonish_logger()


def verify(
    img1_path: Union[str, np.ndarray, List[float]],
    img2_path: Union[str, np.ndarray, List[float]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons.

    The verification function converts facial images to vectors and calculates the similarity
    between those vectors. Vectors of images of the same person should exhibit higher similarity
    (or lower distance) than vectors of images of different persons.

    Args:
        img1_path (str or np.ndarray or List[float]): Path to the first image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        img2_path (str or np.ndarray or  or List[float]): Path to the second image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv)

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        result (dict): A dictionary containing verification results.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity.

        - 'threshold' (float): The maximum threshold used for verification.
            If the distance is below this threshold, the images are considered a match.

        - 'model' (str): The chosen face recognition model.

        - 'similarity_metric' (str): The chosen similarity metric for measuring distances.

        - 'facial_areas' (dict): Rectangular regions of interest for faces in both images.
            - 'img1': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the first image.
            - 'img2': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the second image.

        - 'time' (float): Time taken for the verification process in seconds.
    """

    tic = time.time()

    model: FacialRecognition = modeling.build_model(model_name)
    dims = model.output_shape

    # extract faces from img1
    if isinstance(img1_path, list):
        # given image is already pre-calculated embedding
        if not all(isinstance(dim, float) for dim in img1_path):
            raise ValueError(
                "When passing img1_path as a list, ensure that all its items are of type float."
            )

        if silent is False:
            logger.warn(
                "You passed 1st image as pre-calculated embeddings."
                f"Please ensure that embeddings have been calculated for the {model_name} model."
            )

        if len(img1_path) != dims:
            raise ValueError(
                f"embeddings of {model_name} should have {dims} dimensions,"
                f" but it has {len(img1_path)} dimensions input"
            )

        img1_embeddings = [img1_path]
        img1_facial_areas = [None]
    else:
        try:
            img1_embeddings, img1_facial_areas = __extract_faces_and_embeddings(
                img_path=img1_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
                normalization=normalization,
                anti_spoofing=anti_spoofing,
            )
        except ValueError as err:
            raise ValueError("Exception while processing img1_path") from err

    # extract faces from img2
    if isinstance(img2_path, list):
        # given image is already pre-calculated embedding
        if not all(isinstance(dim, float) for dim in img2_path):
            raise ValueError(
                "When passing img2_path as a list, ensure that all its items are of type float."
            )

        if silent is False:
            logger.warn(
                "You passed 2nd image as pre-calculated embeddings."
                f"Please ensure that embeddings have been calculated for the {model_name} model."
            )

        if len(img2_path) != dims:
            raise ValueError(
                f"embeddings of {model_name} should have {dims} dimensions,"
                f" but it has {len(img2_path)} dimensions input"
            )

        img2_embeddings = [img2_path]
        img2_facial_areas = [None]
    else:
        try:
            img2_embeddings, img2_facial_areas = __extract_faces_and_embeddings(
                img_path=img2_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
                normalization=normalization,
                anti_spoofing=anti_spoofing,
            )
        except ValueError as err:
            raise ValueError("Exception while processing img2_path") from err

    no_facial_area = {
        "x": None,
        "y": None,
        "w": None,
        "h": None,
        "left_eye": None,
        "right_eye": None,
    }

    distances = []
    facial_areas = []
    for idx, img1_embedding in enumerate(img1_embeddings):
        for idy, img2_embedding in enumerate(img2_embeddings):
            distance = find_distance(img1_embedding, img2_embedding, distance_metric)
            distances.append(distance)
            facial_areas.append(
                (img1_facial_areas[idx] or no_facial_area, img2_facial_areas[idy] or no_facial_area)
            )

    # find the face pair with minimum distance
    threshold = threshold or find_threshold(model_name, distance_metric)
    distance = float(min(distances))  # best distance
    facial_areas = facial_areas[np.argmin(distances)]

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
        "time": round(toc - tic, 2),
    }

    return resp_obj


def __extract_faces_and_embeddings(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
) -> Tuple[List[List[float]], List[dict]]:
    """
    Extract facial areas and find corresponding embeddings for given image
    Returns:
        embeddings (List[float])
        facial areas (List[dict])
    """
    embeddings = []
    facial_areas = []

    img_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    # find embeddings for each face
    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in given image.")
        img_embedding_obj = representation.represent(
            img_path=img_obj["face"],
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )
        # already extracted face given, safe to access its 1st item
        img_embedding = img_embedding_obj[0]["embedding"]
        embeddings.append(img_embedding)
        facial_areas.append(img_obj["facial_area"])

    return embeddings, facial_areas


def find_cosine_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x: Union[np.ndarray, list]) -> np.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
    Returns:
        y (np.ndarray): l2 normalized vector
    """
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def find_distance(
    alpha_embedding: Union[np.ndarray, list],
    beta_embedding: Union[np.ndarray, list],
    distance_metric: str,
) -> np.float64:
    """
    Wrapper to find distance between vectors according to the given distance metric
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        distance = find_euclidean_distance(
            l2_normalize(alpha_embedding), l2_normalize(beta_embedding)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return distance


def find_threshold(model_name: str, distance_metric: str) -> float:
    """
    Retrieve pre-tuned threshold values for a model and distance metric pair
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        distance_metric (str): distance metric name. Options are cosine, euclidean
            and euclidean_l2.
    Returns:
        threshold (float): threshold value for that model name and distance metric
            pair. Distances less than this threshold will be classified same person.
    """

    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
        "VGG-Face": {
            "cosine": 0.68,
            "euclidean": 1.17,
            "euclidean_l2": 1.17,
        },  # 4096d - tuned with LFW
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
        "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "euclidean_l2": 1.10},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold
