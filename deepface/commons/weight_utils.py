# built-in dependencies
import os
from typing import Optional
import zipfile
import bz2

# 3rd party dependencies
import gdown

# project dependencies
from deepface.commons import folder_utils, package_utils
from deepface.commons.logger import Logger

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Sequential
else:
    from tensorflow.keras.models import Sequential

logger = Logger()

# pylint: disable=line-too-long
WEIGHTS = {
    "facial_recognition": {
        "VGG-Face": "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5",
        "Facenet": "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5",
        "Facenet512": "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
        "OpenFace": "https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5",
        "FbDeepFace": "https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip",
        "ArcFace": "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5",
        "DeepID": "https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5",
        "SFace": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "GhostFaceNet": "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5",
        "Dlib": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    },
    "demography": {
        "Age": "https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5",
        "Gender": "https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
        "Emotion": "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
        "Race": "https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
    },
    "detection": {
        "ssd_model": "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt",
        "ssd_weights": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "yolo": "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb",
        "yunet": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "dlib": "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
        "centerface": "https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx",
    },
    "spoofing": {
        "MiniFASNetV2": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        "MiniFASNetV1SE": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
    },
}

ALLOWED_COMPRESS_TYPES = ["zip", "bz2"]


def download_weights_if_necessary(
    file_name: str, source_url: str, compress_type: Optional[str] = None
) -> str:
    """
    Download the weights of a pre-trained model from external source if not downloaded yet.
    Args:
        file_name (str): target file name with extension
        source_url (url): source url to be downloaded
        compress_type (optional str): compress type e.g. zip or bz2
    Returns
        target_file (str): exact path for the target file
    """
    home = folder_utils.get_deepface_home()

    target_file = os.path.normpath(os.path.join(home, ".deepface/weights", file_name))

    if os.path.isfile(target_file):
        logger.debug(f"{file_name} is already available at {target_file}")
        return target_file

    if compress_type is not None and compress_type not in ALLOWED_COMPRESS_TYPES:
        raise ValueError(f"unimplemented compress type - {compress_type}")

    try:
        logger.info(f"🔗 {file_name} will be downloaded from {source_url} to {target_file}...")

        if compress_type is None:
            gdown.download(source_url, target_file, quiet=False)
        elif compress_type is not None and compress_type in ALLOWED_COMPRESS_TYPES:
            gdown.download(source_url, f"{target_file}.{compress_type}", quiet=False)

    except Exception as err:
        raise ValueError(
            f"⛓️‍💥 An exception occurred while downloading {file_name} from {source_url}. "
            f"Consider downloading it manually to {target_file}."
        ) from err

    # uncompress downloaded file
    if compress_type == "zip":
        with zipfile.ZipFile(f"{target_file}.zip", "r") as zip_ref:
            zip_ref.extractall(os.path.join(home, ".deepface/weights"))
            logger.info(f"{target_file}.zip unzipped")
    elif compress_type == "bz2":
        bz2file = bz2.BZ2File(f"{target_file}.bz2")
        data = bz2file.read()
        with open(target_file, "wb") as f:
            f.write(data)
        logger.info(f"{target_file}.bz2 unzipped")

    return target_file


def load_model_weights(model: Sequential, weight_file: str) -> Sequential:
    """
    Load pre-trained weights for a given model
    Args:
        model (keras.models.Sequential): pre-built model
        weight_file (str): exact path of pre-trained weights
    Returns:
        model (keras.models.Sequential): pre-built model with
            updated weights
    """
    try:
        model.load_weights(weight_file)
    except Exception as err:
        raise ValueError(
            f"An exception occurred while loading the pre-trained weights from {weight_file}."
            "This might have happened due to an interruption during the download."
            "You may want to delete it and allow DeepFace to download it again during the next run."
            "If the issue persists, consider downloading the file directly from the source "
            "and copying it to the target folder."
        ) from err
    return model


def retrieve_model_source(model_name: str, task: str) -> str:
    """
    Find the source url of a given model name
    Args:
        model_name (str): given model name
    Returns:
        weight_url (str): source url of the given model
    """
    if task not in ["facial_recognition", "detection", "demography", "spoofing"]:
        raise ValueError(f"unimplemented task - {task}")

    source_url = WEIGHTS.get(task, {}).get(model_name)
    if source_url is None:
        raise ValueError(f"Source url cannot be found for given model {task}-{model_name}")
    return source_url
