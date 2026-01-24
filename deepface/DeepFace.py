# common dependencies
import os
import warnings
import logging
from typing import Any, Dict, IO, List, Union, Optional, Sequence, Tuple, cast

# this has to be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# pylint: disable=wrong-import-position, too-many-positional-arguments

# 3rd party dependencies
from numpy.typing import NDArray
import pandas as pd
import tensorflow as tf
from lightphe import LightPHE
from lightdsa import LightDSA

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
    preprocessing,
    datastore,
)
from deepface import __version__

logger = Logger()

# -----------------------------------
# configurations for dependencies

# users should install tf_keras package if they are using tf 2.16 or later versions
package_utils.validate_for_keras3()

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------

# create required folders if necessary to store model weights
folder_utils.initialize_folder()


def build_model(model_name: str, task: str = "facial_recognition") -> Any:
    """
    This function builds a pre-trained model
    Args:
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace GhostFaceNet and Buffalo_L for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe,
                yolov8n, yolov8m, yolov8l, yolov11n, yolov11s, yolov11m,
                yolov11l, yolov12n, yolov12s, yolov12m, yolov12l,
                yunet, fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
            default is facial_recognition
    Returns:
        built_model
    """
    return modeling.build_model(task=task, model_name=model_name)


def verify(
    img1_path: Union[str, NDArray[Any], IO[bytes], List[float]],
    img2_path: Union[str, NDArray[Any], IO[bytes], List[float]],
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
    Args:
        img1_path (str or np.ndarray or IO[bytes] or List[float]): Path to the first image.
            Accepts exact image path as a string, numpy array (BGR), a file object that supports
            at least `.read` and is opened in binary mode, base64 encoded images
            or pre-calculated embeddings.

        img2_path (str or np.ndarray or IO[bytes] or List[float]): Path to the second image.
            Accepts exact image path as a string, numpy array (BGR), a file object that supports
            at least `.read` and is opened in binary mode, base64 encoded images
            or pre-calculated embeddings.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip' (default is opencv).

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
        result (dict): A dictionary containing verification results with following keys.

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

        - 'distance_metric' (str): The chosen similarity metric for measuring distances.

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
        silent=silent,
        threshold=threshold,
        anti_spoofing=anti_spoofing,
    )


def analyze(
    img_path: Union[str, NDArray[Any], IO[bytes], List[str], List[NDArray[Any]], List[IO[bytes]]],
    actions: Union[Tuple[str, ...], List[str]] = ("emotion", "age", "gender", "race"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Analyze facial attributes such as age, gender, emotion, and race in the provided image.
    Args:
        img_path (str, np.ndarray, IO[bytes], list): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If the source image contains multiple faces,
            the result will include information for each detected face.

        actions (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n',  'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2', 'angular' (default is cosine).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        (List[List[Dict[str, Any]]]): A list of analysis results if received batched image,
                                      explained below.

        (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
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
        anti_spoofing=anti_spoofing,
    )


def find(
    img_path: Union[str, NDArray[Any], IO[bytes]],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    similarity_search: bool = False,
    k: Optional[int] = None,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
    refresh_database: bool = True,
    anti_spoofing: bool = False,
    batched: bool = False,
    credentials: Optional[Union[LightDSA, Dict[str, Any]]] = None,
) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:
    """
    Identify individuals in a database. This is a stateful facial recognition function.
        Use search function to do it in a stateless way.
    Args:
        img_path (str or np.ndarray or IO[bytes]): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If the source image contains multiple
            faces, the result will include information for each detected face.

        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2', 'angular' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        similarity_search (boolean): If False, performs identity verification and returns images of
            the same person. If True, performs similarity search and returns visually similar faces
            (e.g., celebrity or parental look-alikes). Default is False.

        k (int): Number of top similar faces to retrieve from the database for each detected face.
            If not specified, all faces within the threshold will be returned (default is None).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        refresh_database (boolean): Synchronizes the images representation (pkl) file with the
            directory/db files, if set to false, it will ignore any file changes inside the db_path
            (default is True).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        credentials (LightDSA or dict): public - private key pair. This will be used to sign
            and verify the integrity of the datastore pickle file. Since pickle files are not safe
            to load from untrusted sources, signing helps detect tampering and prevents loading a
            modified datastore that could execute arbitrary code.

            ```
            from lightdsa import LightDSA
            cs = LightDSA(algorithm_name = "eddsa")
            DeepFace.find(..., credentials=cs)
            # DeepFace.find(..., credentials={**cs.dsa.keys, "algorithm_name": cs.algorithm_name})
            ```

            See LightDSA repo for more details: https://github.com/serengil/LightDSA

    Returns:
        results (List[pd.DataFrame] or List[List[Dict[str, Any]]]):
            A list of pandas dataframes (if `batched=False`) or
            a list of dicts (if `batched=True`).
            Each dataframe or dict corresponds to the identity information for
            an individual detected in the source image.

            Note: If you have a large database and/or a source photo with many faces,
            use `batched=True`, as it is optimized for large batch processing.
            Please pay attention that when using `batched=True`, the function returns
            a list of dicts (not a list of DataFrames),
            but with the same keys as the columns in the DataFrame.

            The DataFrame columns or dict keys include:

            - 'identity': Identity label of the detected individual.

            - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
                    target face in the database.

            - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
                    detected face in the source image.

            - 'threshold': threshold to determine a pair whether same person or different persons

            - 'distance': Similarity score between the faces based on the
                    specified model and distance metric

            - 'confidence': Confidence score indicating the likelihood that the images
                    represent the same person. The score is between 0 and 100, where higher values
                    indicate greater confidence in the verification result.
    """
    return recognition.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        similarity_search=similarity_search,
        k=k,
        expand_percentage=expand_percentage,
        threshold=threshold,
        normalization=normalization,
        silent=silent,
        refresh_database=refresh_database,
        anti_spoofing=anti_spoofing,
        batched=batched,
        credentials=credentials,
    )


def represent(
    img_path: Union[str, NDArray[Any], IO[bytes], Sequence[Union[str, NDArray[Any], IO[bytes]]]],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
    l2_normalize: bool = False,
    minmax_normalize: bool = False,
    return_face: bool = False,
    cryptosystem: Optional[LightPHE] = None,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str, np.ndarray, IO[bytes], or Sequence[Union[str, np.ndarray, IO[bytes]]]):
            The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If the source image contains multiple faces,
            the result will include information for each detected face. If a sequence is provided,
            each element should be a string or numpy array representing an image, and the function
            will process images in batch.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet
            (default is VGG-Face.).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images
            (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
            (default is base).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed (default is None).

        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
            of the output embeddings

        minmax_normalize (bool): Flag to enable min-max normalization of the output embeddings
            to the range [0, 1].

        return_face (bool): If True, the detected face images will also be returned along
            with embeddings. Default is False.

        cryptosystem (LightPHE): An instance of a partially homomorphic encryption system
            to encrypt the output embeddings. If provided, the embeddings will be encrypted
            using the specified cryptosystem. Then, you will be able to perform homomorphic
            operations on the encrypted embeddings without decrypting them first.
            Check out the repo to find out more: https://github.com/serengil/lightphe

    Returns:
        results (List[Dict[str, Any]] or List[Dict[str, Any]]): A list of dictionaries.
            Result type becomes List of List of Dict if batch input passed.
            Each containing the following fields:

        - embedding (List[float]): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).

        - facial_area (dict): Detected facial area by face detection in dictionary format.
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.

        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.

        - encrypted_embedding (List[Any]): Encrypted multidimensional vector representing
            facial features. This field is included only if a `cryptosystem` is provided.
    """
    return representation.represent(
        img_path=img_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
        l2_normalize=l2_normalize,
        minmax_normalize=minmax_normalize,
        return_face=return_face,
        cryptosystem=cryptosystem,
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
    anti_spoofing: bool = False,
    output_path: Optional[str] = None,
    debug: bool = False,
) -> None:
    """
    Run real time face recognition and facial attribute analysis

    Args:
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2', 'angular' (default is cosine).

        enable_face_analysis (bool): Flag to enable face analysis (default is True).

        source (Any): The source for the video stream (default is 0, which represents the
            default camera).

        time_threshold (int): The time threshold (in seconds) for face recognition (default is 5).

        frame_threshold (int): The frame threshold for face recognition (default is 5).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        output_path (str): Path to save the output video. (default is None
            If None, no video is saved).

        debug (bool): set this to True to save frame outcomes

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
        anti_spoofing=anti_spoofing,
        output_path=output_path,
        debug=debug,
    )


def extract_faces(
    img_path: Union[str, NDArray[Any], IO[bytes], List[str], List[NDArray[Any]], List[IO[bytes]]],
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False,
    color_face: str = "rgb",
    normalize_face: bool = True,
    anti_spoofing: bool = False,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Extract faces from a given image

    Args:
        img_path (str or list of str ornp.ndarray or IO[bytes]): Path to the first image.
            Accepts exact image path as a string, list of string, numpy array (BGR), a file object
            that supports at least `.read` and is opened in binary mode, or base64 encoded images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        grayscale (boolean): (Deprecated) Flag to convert the output face image to grayscale
            (default is False).

        color_face (string): Color to return face image output. Options: 'rgb', 'bgr' or 'gray'
            (default is 'rgb').

        normalize_face (boolean): Flag to enable normalization (divide by 255) of the output
            face image output face image normalization (default is True).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (np.ndarray): The detected face as a NumPy array.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values. left and right eyes
                are eyes on the left and right respectively with respect to the person itself
                instead of observer.

        - "confidence" (float): The confidence score associated with the detected face.

        - "is_real" (boolean): antispoofing analyze result. this key is just available in the
            result only if anti_spoofing is set to True in input arguments.

        - "antispoof_score" (float): score of antispoofing analyze result. this key is
            just available in the result only if anti_spoofing is set to True in input arguments.
    """

    return detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        grayscale=grayscale,
        color_face=color_face,
        normalize_face=normalize_face,
        anti_spoofing=anti_spoofing,
    )


def cli() -> None:
    """
    command line interface function will be offered in this block
    """
    import fire

    fire.Fire()


# deprecated function(s)


def detectFace(
    img_path: Union[str, NDArray[Any], IO[bytes]],
    target_size: Tuple[int, int] = (224, 224),
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
) -> Union[NDArray[Any], None]:
    """
    Deprecated face detection function. Use extract_faces for same functionality.

    Args:
        img_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image (default is (224, 224)).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

    Returns:
        img (np.ndarray): detected (and aligned) facial area image as numpy array
    """
    logger.warn("Function detectFace is deprecated. Use extract_faces instead.")
    face_objs: List[Dict[str, Any]] = cast(
        List[Dict[str, Any]],
        extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        ),
    )
    extracted_face = None
    if len(face_objs) > 0:
        extracted_face = face_objs[0]["face"]
        extracted_face = preprocessing.resize_image(img=extracted_face, target_size=target_size)
    return extracted_face


def register(
    img: Union[str, NDArray[Any], IO[bytes], List[str], List[NDArray[Any]], List[IO[bytes]]],
    img_name: Optional[str] = None,
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    l2_normalize: bool = False,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    database_type: str = "postgres",
    connection_details: Optional[Union[Dict[str, Any], str]] = None,
    connection: Any = None,
) -> Dict[str, Any]:
    """
    Register identities to database for face recognition
    Args:
        img (str or np.ndarray or IO[bytes] or list): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If a list is provided, each element should be a string
            or numpy array representing an image, and the function will process images in batch.
        img_name (optional str): image name to store in db, if not provided then we will try to
            extract it from given img.
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip' (default is opencv).
        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).
        align (bool): Flag to enable face alignment (default is True).
        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
        expand_percentage (int): expand detected facial area with a percentage (default is 0).
        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).
        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).
        database_type (str): Type of database to register identities. Options: 'postgres', 'mongo',
            'weaviate', 'neo4j', 'pgvector', 'pinecone' (default is 'postgres').
        connection_details (dict or str): Connection details for the database.
        connection (Any): Existing database connection object. If provided, this connection
            will be used instead of creating a new one.

        Note:
            Instead of providing `connection` or `connection_details`, database connection
            information can be supplied via environment variables:
            - DEEPFACE_POSTGRES_URI
            - DEEPFACE_MONGO_URI
            - DEEPFACE_WEAVIATE_URI
            - DEEPFACE_NEO4J_URI
            - DEEPFACE_PINECONE_API_KEY
    Returns:
        result (dict): A dictionary containing registration results with following keys.
            - inserted (int): Number of embeddings successfully registered to the database.
    """
    return datastore.register(
        img=img,
        img_name=img_name,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        l2_normalize=l2_normalize,
        expand_percentage=expand_percentage,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
        database_type=database_type,
        connection_details=connection_details,
        connection=connection,
    )


def search(
    img: Union[str, NDArray[Any], IO[bytes], List[str], List[NDArray[Any]], List[IO[bytes]]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    l2_normalize: bool = False,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    similarity_search: bool = False,
    k: Optional[int] = None,
    database_type: str = "postgres",
    connection_details: Optional[Union[Dict[str, Any], str]] = None,
    connection: Any = None,
    search_method: str = "exact",
) -> List[pd.DataFrame]:
    """
    Search for identities in database for face recognition. This is a stateless facial
        recognition function. Use find function to do it in a stateful way.
    Args:
        img (str or np.ndarray or IO[bytes] or list): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If a list is provided, each element should be a string
            or numpy array representing an image, and the function will process images in batch.
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip' (default is opencv).
        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'angular' (default is cosine).
        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).
        align (bool): Flag to enable face alignment (default is True).
        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
        expand_percentage (int): expand detected facial area with a percentage (default is 0).
        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).
        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).
        similarity_search (boolean): If False, performs identity verification and returns images of
            the same person. If True, performs similarity search and returns visually similar faces
            (e.g., celebrity or parental look-alikes). Default is False.
        k (int): Number of top similar faces to retrieve from the database for each detected face.
            If not specified, all faces within the threshold will be returned (default is None).
        search_method (str): Method to use for searching identities. Options: 'exact', 'ann'.
            To use ann search, you must run build_index function first to create the index.
        database_type (str): Type of database to search identities. Options: 'postgres', 'mongo',
            'weaviate', 'neo4j', 'pgvector', 'pinecone' (default is 'postgres').
        connection_details (dict or str): Connection details for the database.
        connection (Any): Existing database connection object. If provided, this connection
            will be used instead of creating a new one.

        Note:
            Instead of providing `connection` or `connection_details`, database connection
            information can be supplied via environment variables:
            - DEEPFACE_POSTGRES_URI
            - DEEPFACE_MONGO_URI
            - DEEPFACE_WEAVIATE_URI
            - DEEPFACE_NEO4J_URI
            - DEEPFACE_PINECONE_API_KEY
    Returns:
        results (List[pd.DataFrame]):
            A list of pandas dataframes or a list of dicts. Each dataframe or dict corresponds
                to the identity information for an individual detected in the source image.

            The DataFrame columns or dict keys include:
            - id: ID of the detected individual.
            - img_name: Name of the image file in the database.
            - model_name: Name of the model used for recognition.
            - aligned: Whether face alignment was performed.
            - l2_normalized: Whether L2 normalization was applied.
            - search_method: Method used for searching identities: exact or ann.
            - confidence: Confidence score indicating the likelihood that the images
                represent the same person. The score is between 0 and 100, where higher values
                indicate greater confidence in the verification result.
            - target_x, target_y, target_w, target_h: Bounding box coordinates of the
                target face in the database. Notice that source image's face coordinates
                are not included in the result here.
            - threshold: threshold to determine a pair whether same person or different persons
            - distance_metric: Distance metric used for similarity measurement.
                Distance metric will be ignored for ann search, and set to cosine if l2_normalize
                is True, euclidean if l2_normalize is False.
            - distance: Similarity score between the faces based on the specified model
                and distance metric
    """
    return datastore.search(
        img=img,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        l2_normalize=l2_normalize,
        expand_percentage=expand_percentage,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
        similarity_search=similarity_search,
        k=k,
        database_type=database_type,
        connection_details=connection_details,
        connection=connection,
        search_method=search_method,
    )


def build_index(
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    align: bool = True,
    l2_normalize: bool = False,
    database_type: str = "postgres",
    connection: Any = None,
    connection_details: Optional[Union[Dict[str, Any], str]] = None,
    max_neighbors_per_node: int = 32,
) -> None:
    """
    Build index for faster search in the database. You should set search_method to 'ann'
        in the search function to use the built index.

    - Use this function after registering all identities to the database.
    - This function is resumable, run again whenever new identities are added to the db.
    - Vector databases handle indexing internally, so you don't need to use this function
        when using a vector database ('weaviate', 'neo4j', 'pgvector', 'pinecone')
        as database_type.

    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip' (default is opencv).
        align (bool): Flag to enable face alignment (default is True).
        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
        max_neighbors_per_node (int): Maximum number of neighbors per node in the index
            (default is 32).
        database_type (str): Type of database to build index. Options: 'postgres', 'mongo',
            'weaviate', 'neo4j', 'pgvector', 'pinecone' (default is 'postgres').
        connection (Any): Existing database connection object. If provided, this connection
            will be used instead of creating a new one.
        connection_details (dict or str): Connection details for the database.

        Note:
            Instead of providing `connection` or `connection_details`, database connection
            information can be supplied via environment variables:
            - DEEPFACE_POSTGRES_URI
            - DEEPFACE_MONGO_URI
            - DEEPFACE_WEAVIATE_URI
            - DEEPFACE_NEO4J_URI
            - DEEPFACE_PINECONE_API_KEY
    """
    return datastore.build_index(
        model_name=model_name,
        detector_backend=detector_backend,
        l2_normalize=l2_normalize,
        align=align,
        database_type=database_type,
        connection=connection,
        connection_details=connection_details,
        max_neighbors_per_node=max_neighbors_per_node,
    )
