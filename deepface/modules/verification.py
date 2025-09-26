# built-in dependencies
import time
from typing import Any, Dict, Optional, Union, List, Tuple
import math

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules import representation, detection, modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger
from deepface.config.confidence import confidences
from deepface.config.threshold import thresholds


logger = Logger()


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
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l'
            'centerface' or 'skip' (default is opencv)

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2', 'angular' (default is cosine).

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

        - 'confidence' (float): Confidence score indicating the likelihood that the images
            represent the same person. The score is between 0 and 100, where higher values
            indicate greater confidence in the verification result.

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

    model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name=model_name
    )
    dims = model.output_shape

    no_facial_area = {
        "x": None,
        "y": None,
        "w": None,
        "h": None,
        "left_eye": None,
        "right_eye": None,
    }

    def extract_embeddings_and_facial_areas(
        img_path: Union[str, np.ndarray, List[float]], index: int
    ) -> Tuple[List[List[float]], List[dict]]:
        """
        Extracts facial embeddings and corresponding facial areas from an
        image or returns pre-calculated embeddings.

        Depending on the type of img_path, the function either extracts
        facial embeddings from the provided image
        (via a path or NumPy array) or verifies that the input is a list of
        pre-calculated embeddings and validates them.

        Args:
            img_path (Union[str, np.ndarray, List[float]]):
                - A string representing the file path to an image,
                - A NumPy array containing the image data,
                - Or a list of pre-calculated embedding values (of type `float`).
            index (int): An index value used in error messages and logging
            to identify the number of the image.

        Returns:
            Tuple[List[List[float]], List[dict]]:
                - A list containing lists of facial embeddings for each detected face.
                - A list of dictionaries where each dictionary contains facial area information.
        """
        if isinstance(img_path, list):
            # given image is already pre-calculated embedding
            if not all(isinstance(dim, (float, int)) for dim in img_path):

                raise ValueError(
                    f"When passing img{index}_path as a list,"
                    " ensure that all its items are of type float."
                )

            if silent is False:
                logger.warn(
                    f"You passed {index}-th image as pre-calculated embeddings."
                    "Please ensure that embeddings have been calculated"
                    f" for the {model_name} model."
                )

            if len(img_path) != dims:
                raise ValueError(
                    f"embeddings of {model_name} should have {dims} dimensions,"
                    f" but {index}-th image has {len(img_path)} dimensions input"
                )

            img_embeddings = [img_path]
            img_facial_areas = [no_facial_area]
        else:
            try:
                img_embeddings, img_facial_areas = __extract_faces_and_embeddings(
                    img_path=img_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                    align=align,
                    expand_percentage=expand_percentage,
                    normalization=normalization,
                    anti_spoofing=anti_spoofing,
                )
            except ValueError as err:
                raise ValueError(f"Exception while processing img{index}_path") from err
        return img_embeddings, img_facial_areas

    img1_embeddings, img1_facial_areas = extract_embeddings_and_facial_areas(img1_path, 1)
    img2_embeddings, img2_facial_areas = extract_embeddings_and_facial_areas(img2_path, 2)

    min_distance, min_idx, min_idy = float("inf"), None, None
    for idx, img1_embedding in enumerate(img1_embeddings):
        for idy, img2_embedding in enumerate(img2_embeddings):
            distance = find_distance(img1_embedding, img2_embedding, distance_metric)
            if distance < min_distance:
                min_distance, min_idx, min_idy = distance, idx, idy

    # find the face pair with minimum distance
    pretuned_threshold = find_threshold(model_name, distance_metric)
    threshold = threshold or pretuned_threshold
    distance = float(min_distance)
    confidence = find_confidence(
        distance=distance,
        model_name=model_name,
        distance_metric=distance_metric,
        verified=distance <= pretuned_threshold,
    )
    facial_areas = (
        no_facial_area if min_idx is None else img1_facial_areas[min_idx],
        no_facial_area if min_idy is None else img2_facial_areas[min_idy],
    )

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "confidence": confidence,
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
            img_path=img_obj["face"][:, :, ::-1],  # make compatible with direct representation call
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
) -> Union[np.float64, np.ndarray]:
    """
    Find cosine distance between two given vectors or batches of vectors.
    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.
    Returns
        np.float64 or np.ndarray: Calculated cosine distance(s).
        It returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    if source_representation.ndim == 1 and test_representation.ndim == 1:
        # single embedding
        dot_product = np.dot(source_representation, test_representation)
        source_norm = np.linalg.norm(source_representation)
        test_norm = np.linalg.norm(test_representation)
        distances = 1 - dot_product / (source_norm * test_norm)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        # list of embeddings (batch)
        source_normed = l2_normalize(source_representation, axis=1)  # (N, D)
        test_normed = l2_normalize(test_representation, axis=1)  # (M, D)
        cosine_similarities = np.dot(test_normed, source_normed.T)  # (M, N)
        distances = 1 - cosine_similarities
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    return distances


def find_angular_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> Union[np.float64, np.ndarray]:
    """
    Find angular distance between two vectors or batches of vectors.

    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.

    Returns:
        np.float64 or np.ndarray: angular distance(s).
            Returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """

    # calculate cosine similarity first
    # then convert to angular distance
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    if source_representation.ndim == 1 and test_representation.ndim == 1:
        # single embedding
        dot_product = np.dot(source_representation, test_representation)
        source_norm = np.linalg.norm(source_representation)
        test_norm = np.linalg.norm(test_representation)
        similarity = dot_product / (source_norm * test_norm)
        distances = np.arccos(similarity) / np.pi
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        # list of embeddings (batch)
        source_normed = l2_normalize(source_representation, axis=1)  # (N, D)
        test_normed = l2_normalize(test_representation, axis=1)  # (M, D)
        similarity = np.dot(test_normed, source_normed.T)  # (M, N)
        distances = np.arccos(similarity) / np.pi
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    return distances


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> Union[np.float64, np.ndarray]:
    """
    Find Euclidean distance between two vectors or batches of vectors.

    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.

    Returns:
        np.float64 or np.ndarray: Euclidean distance(s).
            Returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    # Single embedding case (1D arrays)
    if source_representation.ndim == 1 and test_representation.ndim == 1:
        distances = np.linalg.norm(source_representation - test_representation)
    # Batch embeddings case (2D arrays)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        diff = (
            source_representation[None, :, :] - test_representation[:, None, :]
        )  # (N, D) - (M, D)  = (M, N, D)
        distances = np.linalg.norm(diff, axis=2)  # (M, N)
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    return distances


def l2_normalize(
    x: Union[np.ndarray, list], axis: Union[int, None] = None, epsilon: float = 1e-10
) -> np.ndarray:
    """
    Normalize input vector with l2
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


def find_distance(
    alpha_embedding: Union[np.ndarray, list],
    beta_embedding: Union[np.ndarray, list],
    distance_metric: str,
) -> Union[np.float64, np.ndarray]:
    """
    Wrapper to find the distance between vectors based on the specified distance metric.

    Args:
        alpha_embedding (np.ndarray or list): 1st vector or batch of vectors.
        beta_embedding (np.ndarray or list): 2nd vector or batch of vectors.
        distance_metric (str): The type of distance to compute
            ('cosine', 'euclidean', 'euclidean_l2', or 'angular').

    Returns:
        np.float64 or np.ndarray: The calculated distance(s).
    """
    # Convert inputs to numpy arrays if necessary
    alpha_embedding = np.asarray(alpha_embedding)
    beta_embedding = np.asarray(beta_embedding)

    # Ensure that both embeddings are either 1D or 2D
    if alpha_embedding.ndim != beta_embedding.ndim or alpha_embedding.ndim not in (1, 2):
        raise ValueError(
            f"Both embeddings must be either 1D or 2D, but received "
            f"alpha shape: {alpha_embedding.shape}, beta shape: {beta_embedding.shape}"
        )

    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "angular":
        distance = find_angular_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        axis = None if alpha_embedding.ndim == 1 else 1
        normalized_alpha = l2_normalize(alpha_embedding, axis=axis)
        normalized_beta = l2_normalize(beta_embedding, axis=axis)
        distance = find_euclidean_distance(normalized_alpha, normalized_beta)
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return np.round(distance, 6)


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
    if thresholds.get(model_name) is None:
        raise ValueError(f"Model {model_name} is not supported. ")

    threshold = thresholds.get(model_name, {}).get(distance_metric)

    if threshold is None:
        raise ValueError(
            f"Distance metric {distance_metric} is not available for model {model_name}. "
        )

    return threshold


def find_confidence(
    distance: float, model_name: str, distance_metric: str, verified: bool
) -> float:
    """
    Using pre-built logistic regression model, find confidence value from distance.
        The confidence score provides a probalistic estimate, indicating how likely
        the classification is correct, thus giving softer, more informative measure of
        certainty than a simple binary classification.

        Configuration values are calculated in experiments/distance-to-confidence.ipynb
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        distance_metric (str): distance metric name. Options are cosine, euclidean
            euclidean_l2 and angular.
        verified (bool): True if the images are classified as same person,
            False if different persons.
    Returns:
        confidence (float): confidence value being same person for that model name
            and distance metric pair. Same person classifications confidence should be
            distributed between 51-100% and different person classifications confidence
            should be distributed between 0-49%. The higher the confidence, the more
            certain the model is about the classification.
    """

    if confidences.get(model_name) is None:
        return 51 if verified else 49

    config = confidences[model_name].get(distance_metric)

    if config is None:
        return 51 if verified else 49

    w = config["w"]
    b = config["b"]

    normalizer = config["normalizer"]

    denorm_max_true = config["denorm_max_true"]
    denorm_min_true = config["denorm_min_true"]
    denorm_max_false = config["denorm_max_false"]
    denorm_min_false = config["denorm_min_false"]

    if normalizer > 1:
        distance = distance / normalizer

    z = w * distance + b
    confidence = 100 * (1 / (1 + math.exp(-z)))

    # re-distribute the confidence between 0-49 for different persons, 51-100 for same persons
    if verified:
        min_original = denorm_min_true
        max_original = denorm_max_true
        min_target = max(51, min_original)
        max_target = 100
    else:
        min_original = denorm_min_false
        max_original = denorm_max_false
        min_target = 0
        max_target = min(49, max_original)

    confidence_distributed = ((confidence - min_original) / (max_original - min_original)) * (
        max_target - min_target
    ) + min_target

    # ensure confidence is within 51-100 for same persons and 0-49 for different persons
    if verified and confidence_distributed < 51:
        confidence_distributed = 51
    elif not verified and confidence_distributed > 49:
        confidence_distributed = 49

    # ensure confidence is within 0-100
    if confidence_distributed < 0:
        confidence_distributed = 0
    elif confidence_distributed > 100:
        confidence_distributed = 100

    return round(confidence_distributed, 2)
