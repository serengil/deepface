# common dependencies
import os
import warnings
import logging
from typing import Any, Dict, List, Tuple, Union, Optional

# 3rd party dependencies
import numpy as np
import pandas as pd
import tensorflow as tf

# package dependencies
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.modules import (
    modeling,
    representation,
    verification,
    recognition,
    demography,
    detection,
    streaming,
)
from deepface import __version__

logger = Logger(module="DeepFace")

# -----------------------------------
# configurations for dependencies

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------

# create required folders if necessary to store model weights
folder_utils.initialize_folder()


def build_model(model_name: str) -> Any:
    """
    This function builds a deepface model
    Args:
        model_name (string): face recognition or facial attribute model
            VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
            Age, Gender, Emotion, Race for facial attributes
    Returns:
        built_model
    """
    return modeling.build_model(model_name=model_name)


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons.
    Args:
        img1_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        img2_path (str or np.ndarray): Path to the second image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

    Returns:
        result (dict): A dictionary containing verification results with following keys.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity.

        - 'max_threshold_to_verify' (float): The maximum threshold used for verification.
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

    return verification.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
    )


def analyze(
    img_path: Union[str, np.ndarray],
    actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
) -> List[Dict[str, Any]]:
    """
    Analyze facial attributes such as age, gender, emotion, and race in the provided image.
    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        actions (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
           the analysis results for a detected face. Each dictionary in the list contains the
           following keys:

        - 'region' (dict): Represents the rectangular region of the detected face in the image.
            - 'x': x-coordinate of the top-left corner of the face.
            - 'y': y-coordinate of the top-left corner of the face.
            - 'w': Width of the detected face region.
            - 'h': Height of the detected face region.

        - 'age' (float): Estimated age of the detected face.

        - 'face_confidence' (float): Confidence score for the detected face.
            Indicates the reliability of the face detection.

        - 'dominant_gender' (str): The dominant gender in the detected face.
            Either "Man" or "Woman".

        - 'gender' (dict): Confidence scores for each gender category.
            - 'Man': Confidence score for the male gender.
            - 'Woman': Confidence score for the female gender.

        - 'dominant_emotion' (str): The dominant emotion in the detected face.
            Possible values include "sad," "angry," "surprise," "fear," "happy,"
            "disgust," and "neutral"

        - 'emotion' (dict): Confidence scores for each emotion category.
            - 'sad': Confidence score for sadness.
            - 'angry': Confidence score for anger.
            - 'surprise': Confidence score for surprise.
            - 'fear': Confidence score for fear.
            - 'happy': Confidence score for happiness.
            - 'disgust': Confidence score for disgust.
            - 'neutral': Confidence score for neutrality.

        - 'dominant_race' (str): The dominant race in the detected face.
            Possible values include "indian," "asian," "latino hispanic,"
            "black," "middle eastern," and "white."

        - 'race' (dict): Confidence scores for each race category.
            - 'indian': Confidence score for Indian ethnicity.
            - 'asian': Confidence score for Asian ethnicity.
            - 'latino hispanic': Confidence score for Latino/Hispanic ethnicity.
            - 'black': Confidence score for Black ethnicity.
            - 'middle eastern': Confidence score for Middle Eastern ethnicity.
            - 'white': Confidence score for White ethnicity.
    """
    return demography.analyze(
        img_path=img_path,
        actions=actions,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        silent=silent,
    )


def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
) -> List[pd.DataFrame]:
    """
    Identify individuals in a database
    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

    Returns:
        results (List[pd.DataFrame]): A list of pandas dataframes. Each dataframe corresponds
            to the identity information for an individual detected in the source image.
            The DataFrame columns include:

        - 'identity': Identity label of the detected individual.

        - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
                target face in the database.

        - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
                detected face in the source image.

        - 'threshold': threshold to determine a pair whether same person or different persons

        - 'distance': Similarity score between the faces based on the
                specified model and distance metric
    """
    return recognition.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        threshold=threshold,
        normalization=normalization,
        silent=silent,
    )


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
) -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face.).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images
            (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
            (default is base).

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
    return representation.represent(
        img_path=img_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
    )


def stream(
    db_path: str = "",
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enable_face_analysis: bool = True,
    source: Any = 0,
    time_threshold: int = 5,
    frame_threshold: int = 5,
) -> None:
    """
    Run real time face recognition and facial attribute analysis

    Args:
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enable_face_analysis (bool): Flag to enable face analysis (default is True).

        source (Any): The source for the video stream (default is 0, which represents the
            default camera).

        time_threshold (int): The time threshold (in seconds) for face recognition (default is 5).

        frame_threshold (int): The frame threshold for face recognition (default is 5).
    Returns:
        None
    """

    time_threshold = max(time_threshold, 1)
    frame_threshold = max(frame_threshold, 1)

    streaming.analysis(
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enable_face_analysis=enable_face_analysis,
        source=source,
        time_threshold=time_threshold,
        frame_threshold=frame_threshold,
    )


def extract_faces(
    img_path: Union[str, np.ndarray],
    target_size: Optional[Tuple[int, int]] = (224, 224),
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

        target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image (default is (224, 224)).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        grayscale (boolean): Flag to convert the image to grayscale before
            processing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (np.ndarray): The detected face as a NumPy array.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values

        - "confidence" (float): The confidence score associated with the detected face.
    """

    return detection.extract_faces(
        img_path=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        grayscale=grayscale,
        human_readable=True,
    )


def cli() -> None:
    """
    command line interface function will be offered in this block
    """
    import fire

    fire.Fire()


# deprecated function(s)


def detectFace(
    img_path: Union[str, np.ndarray],
    target_size: tuple = (224, 224),
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
) -> Union[np.ndarray, None]:
    """
    Deprecated face detection function. Use extract_faces for same functionality.

    Args:
        img_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image (default is (224, 224)).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

    Returns:
        img (np.ndarray): detected (and aligned) facial area image as numpy array
    """
    logger.warn("Function detectFace is deprecated. Use extract_faces instead.")
    face_objs = extract_faces(
        img_path=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        grayscale=False,
    )
    extracted_face = None
    if len(face_objs) > 0:
        extracted_face = face_objs[0]["face"]
    return extracted_face
