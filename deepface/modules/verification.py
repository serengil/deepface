# built-in dependencies
import time
from typing import IO, Optional, Union, List, Tuple

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules import representation, detection, modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.models.Verification import Verification
from deepface.commons.logger import Logger

logger = Logger()


def verify(
    img1_path: str,
    img2_path: str,
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
) -> Verification:
    """
    Verify if an image pair represents the same person or different persons.
    Args:
        img1_path (str): Path to the first image.

        img2_path (str): Path to the second image.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2', 'angular' (default is cosine).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        Verification: Verification object containing the verification results.
    """

    tic = time.time()
    threshold = threshold or find_threshold(model_name, distance_metric)

    # extract faces
    faces = representation.represent(
        img_path=[img1_path, img2_path],
        model_name=model_name,
        detector_backend=detector_backend,
        anti_spoofing=anti_spoofing,
        max_faces=1,
    )

    # exit if we failed to find one face per image or embeddings are missing
    if len(faces) != 2 or (faces[0].embedding is None) or (faces[1].embedding is None):
        return Verification(
            face1=faces[0] if len(faces) == 1 else None,
            face2=faces[1] if len(faces) == 2 else None,
            verified=False,
            distance=float("inf"),
            threshold=threshold,
            model=model_name,
            detector_backend=detector_backend,
            metric=distance_metric,
            time=time.time() - tic,
        )

    # compare faces embeddings
    distance = find_distance(faces[0].embedding, faces[1].embedding, distance_metric)

    return Verification(
        face1=faces[0],
        face2=faces[1],
        verified=bool(distance <= threshold),
        distance=distance,
        threshold=threshold,
        model=model_name,
        detector_backend=detector_backend,
        metric=distance_metric,
        time=time.time() - tic,
    )

def find_cosine_distance(alpha_embedding: List[float], beta_embedding: List[float]) -> float:
    """
    Find cosine distance between two given vectors or batches of vectors.

    Args:
        alpha_embedding (List[float]): 1st vector.
        beta_embedding (List[float]): 2nd vector.

    Returns
        float: Calculated cosine distance.
    """

    dot_product = np.dot(alpha_embedding, beta_embedding)
    a_norm = np.linalg.norm(alpha_embedding)
    b_norm = np.linalg.norm(beta_embedding)
    return 1 - dot_product / (a_norm * b_norm)

def find_angular_distance(alpha_embedding: List[float], beta_embedding: List[float]) -> float:
    """
    Find angular distance between two vectors or batches of vectors.

    Args:
        alpha_embedding (List[float]): 1st vector.
        beta_embedding (List[float]): 2nd vector.

    Returns:
        float: Calculated angular distance.
    """

    dot_product = np.dot(alpha_embedding, beta_embedding)
    a_norm = np.linalg.norm(alpha_embedding)
    b_norm = np.linalg.norm(beta_embedding)
    similarity = dot_product / (a_norm * b_norm)
    return np.arccos(similarity) / np.pi

def find_euclidean_distance(alpha_embedding: List[float], beta_embedding: List[float]) -> float:
    """
    Find Euclidean distance between two vectors or batches of vectors.

    Args:
        alpha_embedding (List[float]): 1st vector.
        beta_embedding (List[float]): 2nd vector.

    Returns:
        float: Calculated Euclidean distance.
    """
    distance = np.linalg.norm(np.array(alpha_embedding) - np.array(beta_embedding))
    return float(distance)


def l2_normalize(
    x: Union[np.ndarray, list], axis: Union[int, None] = None, epsilon: float = 0
) -> np.ndarray:
    """
    Normalize input vector with l2.

    Args:
        x (np.ndarray or list): given vector
        axis (int): axis along which to normalize
    Returns:
        np.ndarray: l2 normalized vector
    """
    # Convert inputs to numpy arrays if necessary
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + epsilon)


def find_distance(alpha_embedding: List[float], beta_embedding: List[float], distance_metric: str) -> float:
    """
    Wrapper to find the distance between vectors based on the specified distance metric.

    Args:
        alpha_embedding (List[float]): 1st vector.
        beta_embedding (List[float]): 2nd vector.
        distance_metric (str): The type of distance to compute('cosine', 'euclidean', 'euclidean_l2', or 'angular').

    Returns:
        float: The calculated distance.
    """
    # Convert inputs to numpy arrays if necessary
    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "angular":
        distance = find_angular_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        normalized_alpha = l2_normalize(alpha_embedding)
        normalized_beta = l2_normalize(beta_embedding)
        distance = find_euclidean_distance(normalized_alpha.tolist(), normalized_beta.tolist())
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
            euclidean_l2 and angular.
    Returns:
        threshold (float): threshold value for that model name and distance metric
            pair. Distances less than this threshold will be classified same person.
    """

    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75, "angular": 0.37}

    thresholds = {
        # "VGG-Face": {
        # "cosine": 0.40,
        # "euclidean": 0.60,
        # "euclidean_l2": 0.86,
        #  "angular": 0.37
        # }, # 2622d

        "VGG-Face": {
            "cosine": 0.68,
            "euclidean": 1.17,
            "euclidean_l2": 1.17,
            "angular": 0.39,
        },  # 4096d - tuned with LFW
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80, "angular": 0.33},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04, "angular": 0.35},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13, "angular": 0.39},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4, "angular": 0.12},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055, "angular": 0.36},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55, "angular": 0.11},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64, "angular": 0.12},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17, "angular": 0.04},
        "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "euclidean_l2": 1.10, "angular": 0.38},
        "Buffalo_L": {"cosine": 0.55, "euclidean": 0.6, "euclidean_l2": 1.1, "angular": 0.45},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold
