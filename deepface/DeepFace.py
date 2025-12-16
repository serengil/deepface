# common dependencies
import os
import warnings
import logging
from typing import Any, Dict, IO, List, Tuple, Union, Optional

# this has to be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# pylint: disable=wrong-import-position

# 3rd party dependencies
import numpy as np
import pandas as pd
import tensorflow as tf

# package dependencies
from deepface.models.Face import Face
from deepface.models.Verification import Verification
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
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, yolov11n,
              yolov11s, yolov11m, yunet, fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
            default is facial_recognition
    Returns:
        built_model
    """
    return modeling.build_model(task=task, model_name=model_name)


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

    return verification.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        threshold=threshold,
        anti_spoofing=anti_spoofing,
    )


def analyze(
    img_path: Union[str, np.ndarray, IO[bytes], List[str], List[np.ndarray], List[IO[bytes]]],
    actions: list = ["emotion", "age", "gender", "race"],
    detector_backend: str = "opencv",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
    silent: bool = False,
) -> List[Face]:
    """
    Analyze facial attributes such as age, gender, emotion, and race in the provided image.
    Args:
        img_path (str, np.ndarray, IO[bytes], list): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If the source image contains multiple faces,
            the result will include information for each detected face.

        actions (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n',  'yolov11s', 'yolov11m',
            'centerface' (default is opencv).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed and returned (default is None).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

    Returns:
        List[Face.Face]: A list of Face objects, each containing demographic results for detected faces.
    """
    return demography.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        silent=silent,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces
    )


def represent(
    img_path: Union[str, np.ndarray, IO[bytes], List[str], List[np.ndarray], List[IO[bytes]]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> List[Face]:
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

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' (default is opencv).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed and returned (default is None).

    Returns:
        List[Face.Face]: A list of Face objects, each containing embedding results for detected faces.
    """
    return representation.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
    )



def extract_faces(
    img_path: Union[str, np.ndarray, IO[bytes]],
    detector_backend: str = "opencv",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> Tuple[np.ndarray, List[Face]]:
    """
    Extract faces from a given image

    Args:
        img_path (str or np.ndarray or IO[bytes]): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), a file object that supports at least `.read` and is
            opened in binary mode, or base64 encoded images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed and returned (default is None).

    Returns:
        Tuple[np.ndarray, List[Face]]: A tuple containing the original image as a numpy array and a list of Face objects,
            each containing a valid extracted face and related metadata alongside spoofing analysis if enabled.
    """

    return detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
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
    refresh_database: bool = True,
    anti_spoofing: bool = False,
    batched: bool = False,
) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:
    """
    Identify individuals in a database
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

            - 'antispoof_scores' (dict): antispoofing analyze result. This key is
                just available in the result only if anti_spoofing is set to True in input arguments.
                It contains:
                - "spoof_confidence" (float): Confidence score for spoofing.
                - "real_confidence" (float): Confidence score for real face.
                - "uncertainty" (float): Uncertainty score.
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
        refresh_database=refresh_database,
        anti_spoofing=anti_spoofing,
        batched=batched,
    )



def cli() -> None:
    """
    command line interface function will be offered in this block
    """
    import fire

    fire.Fire()
