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
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
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

    thresholds = {
        "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17, "angular": 0.39},
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
    configs = {
        "VGG-Face": {
            "cosine": {
                "w": -6.61254026479904,
                "b": 3.145549102667549,
                "normalizer": 0.979183,
                "denorm_max_true": 88.28484099529868,
                "denorm_min_true": 20.787978733113846,
                "denorm_max_false": 20.556681914175016,
                "denorm_min_false": 3.457832575746455,
            },
            "euclidean": {
                "w": -6.505124425537925,
                "b": 3.8912125488666436,
                "normalizer": 1.399416,
                "denorm_max_true": 76.47012030708339,
                "denorm_min_true": 17.794898757517597,
                "denorm_max_false": 16.443735520230746,
                "denorm_min_false": 6.82484246933999,
            },
            "euclidean_l2": {
                "w": -6.505124425537925,
                "b": 3.8912125488666436,
                "normalizer": 1.399416,
                "denorm_max_true": 76.47012030708339,
                "denorm_min_true": 17.794898757517597,
                "denorm_max_false": 16.443735520230746,
                "denorm_min_false": 6.82484246933999,
            },
            "angular": {
                "w": -5.035753663256314,
                "b": 0.2961597285248822,
                "normalizer": 0.493373,
                "denorm_max_true": 34.230717614698825,
                "denorm_min_true": 16.624506551373642,
                "denorm_max_false": 15.49607745347531,
                "denorm_min_false": 10.080199045102233,
            },
        },
        "Facenet": {
            "cosine": {
                "w": -6.502269165856082,
                "b": 1.679048923097668,
                "normalizer": 1.206694,
                "denorm_max_true": 77.17253153662926,
                "denorm_min_true": 41.790002608273234,
                "denorm_max_false": 20.618350202170916,
                "denorm_min_false": 0.7976712344840693,
            },
            "euclidean": {
                "w": -6.716177467853723,
                "b": 2.790978346203265,
                "normalizer": 18.735288,
                "denorm_max_true": 74.76412617567517,
                "denorm_min_true": 40.4423755909089,
                "denorm_max_false": 25.840858374979504,
                "denorm_min_false": 1.9356150486888306,
            },
            "euclidean_l2": {
                "w": -6.708710331202137,
                "b": 2.9094193067398195,
                "normalizer": 1.553508,
                "denorm_max_true": 75.45756719896039,
                "denorm_min_true": 40.4509428022908,
                "denorm_max_false": 30.555931000001184,
                "denorm_min_false": 2.189644991619842,
            },
            "angular": {
                "w": -6.371147050396505,
                "b": 0.6766460615182355,
                "normalizer": 0.56627,
                "denorm_max_true": 45.802357900723386,
                "denorm_min_true": 24.327312950719133,
                "denorm_max_false": 16.95267765757785,
                "denorm_min_false": 5.063533287198758,
            },
        },
        "Facenet512": {
            "cosine": {
                "w": -5.868251768046832,
                "b": 0.8674084834913275,
                "normalizer": 1.384316,
                "denorm_max_true": 62.19495119465813,
                "denorm_min_true": 40.700922463435305,
                "denorm_max_false": 39.96947059787277,
                "denorm_min_false": 0.6687247052303562,
            },
            "euclidean": {
                "w": -6.742961485008033,
                "b": 2.61949356815704,
                "normalizer": 40.771751,
                "denorm_max_true": 75.20184708761875,
                "denorm_min_true": 23.391566956420913,
                "denorm_max_false": 20.317410180868695,
                "denorm_min_false": 1.5930391784587872,
            },
            "euclidean_l2": {
                "w": -6.76040230078399,
                "b": 3.0057651791652638,
                "normalizer": 1.663921,
                "denorm_max_true": 78.73795056941638,
                "denorm_min_true": 23.492800781101295,
                "denorm_max_false": 21.754795621979017,
                "denorm_min_false": 2.287349921316484,
            },
            "angular": {
                "w": -6.487840122876141,
                "b": 0.9536496647058516,
                "normalizer": 0.625563,
                "denorm_max_true": 52.1215977454314,
                "denorm_min_true": 21.72372703119752,
                "denorm_max_false": 20.734638325222512,
                "denorm_min_false": 4.290546073131373,
            },
        },
        "ArcFace": {
            "cosine": {
                "w": -6.586150423868342,
                "b": 2.2617127737265186,
                "normalizer": 1.22267,
                "denorm_max_true": 82.32544721731948,
                "denorm_min_true": 28.687745855666044,
                "denorm_max_false": 17.210359124614925,
                "denorm_min_false": 1.306796127659645,
            },
            "euclidean": {
                "w": -5.370594347027345,
                "b": 0.6896763568574544,
                "normalizer": 10.401334,
                "denorm_max_true": 43.042326442916284,
                "denorm_min_true": 19.048164749978937,
                "denorm_max_false": 18.682892152590245,
                "denorm_min_false": 0.9185346995033978,
            },
            "euclidean_l2": {
                "w": -6.541948874098929,
                "b": 3.2628963901283585,
                "normalizer": 1.563758,
                "denorm_max_true": 74.93505228104696,
                "denorm_min_true": 21.800168694491585,
                "denorm_max_false": 15.091078907730523,
                "denorm_min_false": 3.6296845343803894,
            },
            "angular": {
                "w": -5.63344888288512,
                "b": 0.5682159222314143,
                "normalizer": 0.571477,
                "denorm_max_true": 40.81365081792457,
                "denorm_min_true": 18.409621072962818,
                "denorm_max_false": 15.13985487535995,
                "denorm_min_false": 6.59169002231532,
            },
        },
        "GhostFaceNet": {
            "cosine": {
                "w": -6.595823624282929,
                "b": 2.4180964556128584,
                "normalizer": 1.155661,
                "denorm_max_true": 84.65757505749636,
                "denorm_min_true": 22.700531204915876,
                "denorm_max_false": 15.08427020899693,
                "denorm_min_false": 1.5101757980158446,
            },
            "euclidean": {
                "w": -6.506709483088198,
                "b": 2.935230928728856,
                "normalizer": 52.150135,
                "denorm_max_true": 73.09358661728027,
                "denorm_min_true": 24.63718600288871,
                "denorm_max_false": 16.513245368123492,
                "denorm_min_false": 2.7345457334612155,
            },
            "euclidean_l2": {
                "w": -6.572852118241916,
                "b": 3.3068829731570824,
                "normalizer": 1.520303,
                "denorm_max_true": 75.95533899593255,
                "denorm_min_true": 20.08886780040665,
                "denorm_max_false": 17.84278657570624,
                "denorm_min_false": 3.675727926946401,
            },
            "angular": {
                "w": -5.588606771251277,
                "b": 0.48845693581021543,
                "normalizer": 0.549751,
                "denorm_max_true": 39.92978715270581,
                "denorm_min_true": 16.49081504576138,
                "denorm_max_false": 16.143233933992967,
                "denorm_min_false": 7.018276839135204,
            },
        },
        "Dlib": {
            "cosine": {
                "w": -3.7618087798972293,
                "b": -0.6908814404797111,
                "normalizer": 0.200732,
                "denorm_max_true": 32.289077089161594,
                "denorm_min_true": 27.84268789569509,
                "denorm_max_false": 27.803329512485526,
                "denorm_min_false": 19.061842082779275,
            },
            "euclidean": {
                "w": -7.494007964438609,
                "b": 3.9696154227506937,
                "normalizer": 0.909856,
                "denorm_max_true": 89.6022157357575,
                "denorm_min_true": 37.63548146988899,
                "denorm_max_false": 36.791885490113046,
                "denorm_min_false": 5.474065355413035,
            },
            "euclidean_l2": {
                "w": -7.104640497107913,
                "b": 2.3288632189212457,
                "normalizer": 0.633612,
                "denorm_max_true": 76.39948218795206,
                "denorm_min_true": 37.84998375132765,
                "denorm_max_false": 36.56103592414769,
                "denorm_min_false": 10.222692344960938,
            },
            "angular": {
                "w": -3.2116944601651727,
                "b": -0.5970349913114966,
                "normalizer": 0.205221,
                "denorm_max_true": 31.79313619700238,
                "denorm_min_true": 27.2527288169969,
                "denorm_max_false": 27.18852931051015,
                "denorm_min_false": 22.163871613552466,
            },
        },
        "SFace": {
            "cosine": {
                "w": -6.531673506288863,
                "b": 2.2217245506456895,
                "normalizer": 1.147363,
                "denorm_max_true": 81.04135659361265,
                "denorm_min_true": 26.26225160110723,
                "denorm_max_false": 23.62218403514688,
                "denorm_min_false": 1.3256149092319998,
            },
            "euclidean": {
                "w": -5.922412546512467,
                "b": 2.2306552793788903,
                "normalizer": 18.826448,
                "denorm_max_true": 53.59440633088357,
                "denorm_min_true": 24.24534043569188,
                "denorm_max_false": 24.000007447174763,
                "denorm_min_false": 2.4321858952665205,
            },
            "euclidean_l2": {
                "w": -6.561644724339029,
                "b": 3.215600266169891,
                "normalizer": 1.514835,
                "denorm_max_true": 72.39437183443313,
                "denorm_min_true": 22.25500614635702,
                "denorm_max_false": 19.531386697831906,
                "denorm_min_false": 3.4024932531839815,
            },
            "angular": {
                "w": -5.575609473928204,
                "b": 0.421773496308517,
                "normalizer": 0.547079,
                "denorm_max_true": 37.48563531943993,
                "denorm_min_true": 17.074673595288292,
                "denorm_max_false": 16.40044990637247,
                "denorm_min_false": 6.732498767352633,
            },
        },
        "OpenFace": {
            "cosine": {
                "w": -2.932562678842587,
                "b": -2.0582304636838518,
                "normalizer": 0.884469,
                "denorm_max_true": 10.546657497687274,
                "denorm_min_true": 8.832401641007316,
                "denorm_max_false": 8.535900401000742,
                "denorm_min_false": 0.9452381276992784,
            },
            "euclidean": {
                "w": -5.829345103053105,
                "b": 1.4390680732209533,
                "normalizer": 1.330015,
                "denorm_max_true": 60.28549643159932,
                "denorm_min_true": 27.506182086242408,
                "denorm_max_false": 27.058382251505012,
                "denorm_min_false": 1.2245483461697035,
            },
            "euclidean_l2": {
                "w": -5.82934526003372,
                "b": 1.4390681633867946,
                "normalizer": 1.330015,
                "denorm_max_true": 60.28549793162577,
                "denorm_min_true": 27.506182591043363,
                "denorm_max_false": 27.058382739158215,
                "denorm_min_false": 1.224548265353685,
            },
            "angular": {
                "w": -0.5288927551138177,
                "b": -4.450627600247528,
                "normalizer": 0.463143,
                "denorm_max_true": 1.109656206367124,
                "denorm_min_true": 1.0914425266009675,
                "denorm_max_false": 1.0887229955132125,
                "denorm_min_false": 0.9052858972210456,
            },
        },
        "DeepID": {
            "cosine": {
                "w": -1.1109389867203003,
                "b": -1.644356005882629,
                "normalizer": 0.142463,
                "denorm_max_true": 16.114369377586183,
                "denorm_min_true": 15.966222060549578,
                "denorm_max_false": 15.96057368926546,
                "denorm_min_false": 14.153198282635255,
            },
            "euclidean": {
                "w": -4.267900227772648,
                "b": -0.11777629548659044,
                "normalizer": 124.194689,
                "denorm_max_true": 27.2342405984107,
                "denorm_min_true": 16.08201206739819,
                "denorm_max_false": 15.90739039527823,
                "denorm_min_false": 1.230125408063358,
            },
            "euclidean_l2": {
                "w": -3.6681410379067394,
                "b": -0.9230555193862335,
                "normalizer": 0.533784,
                "denorm_max_true": 21.681829939622233,
                "denorm_min_true": 17.57160935306287,
                "denorm_max_false": 17.548244679217202,
                "denorm_min_false": 5.309829288484883,
            },
            "angular": {
                "w": -0.2707576106983799,
                "b": -3.87004749575164,
                "normalizer": 0.171993,
                "denorm_max_true": 2.02619851743467,
                "denorm_min_true": 2.0220784432439474,
                "denorm_max_false": 2.021498115793411,
                "denorm_min_false": 1.951974654804089,
            },
        },
        "DeepFace": {
            "cosine": {
                "w": -5.415666808301125,
                "b": 1.4142777745857174,
                "normalizer": 0.379428,
                "denorm_max_true": 74.38935817923884,
                "denorm_min_true": 54.25111593547536,
                "denorm_max_false": 54.16884369236297,
                "denorm_min_false": 34.51159270342188,
            },
            "euclidean": {
                "w": -7.001068623335827,
                "b": 5.229677082777916,
                "normalizer": 85.997482,
                "denorm_max_true": 90.45531547409938,
                "denorm_min_true": 50.79114821321717,
                "denorm_max_false": 50.4218117539695,
                "denorm_min_false": 14.53693626301508,
            },
            "euclidean_l2": {
                "w": -6.520866920809606,
                "b": 3.750706572243227,
                "normalizer": 0.871123,
                "denorm_max_true": 80.42595166943502,
                "denorm_min_true": 39.6419822387709,
                "denorm_max_false": 39.578029519563,
                "denorm_min_false": 12.677597535252195,
            },
            "angular": {
                "w": -0.20178503977417342,
                "b": -4.960267621006636,
                "normalizer": 0.2869,
                "denorm_max_true": 0.6803996880922583,
                "denorm_min_true": 0.6801854995078236,
                "denorm_max_false": 0.6759509854967268,
                "denorm_min_false": 0.6573199053945571,
            },
        },
    }

    if configs.get(model_name) is None:
        return 51 if verified else 49

    config = configs[model_name].get(distance_metric)

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
