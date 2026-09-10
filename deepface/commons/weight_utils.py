# built-in dependencies
import os
from typing import Any, Dict, List, Optional, Union
import zipfile
import bz2

# 3rd party dependencies
import gdown

# project dependencies
from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.modules.exceptions import UnimplementedError

logger = Logger()

# pylint: disable=line-too-long, use-maxsplit-arg

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
        raise UnimplementedError(f"unimplemented compress type - {compress_type}")

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


def load_model_weights(model: Any, weight_file: str) -> Any:
    """
    Load pre-trained weights for a given keras model
    Args:
        model (keras.models.Model): pre-built model
        weight_file (str): exact path of pre-trained weights
    Returns:
        model (keras.models.Model): pre-built model with
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


# where the weight urls of each model live. a model is skipped when it is not
# implemented for the backend engine in use, and the file name is the last part of the
# url unless it is overwritten here
WEIGHT_SOURCES = [
    # facial recognition
    ("facial_recognition", "VGG-Face", ["WEIGHTS_URL"]),
    ("facial_recognition", "Facenet", ["FACENET128_WEIGHTS", "FACENET512_WEIGHTS"]),
    ("facial_recognition", "OpenFace", ["WEIGHTS_URL"]),
    ("facial_recognition", "DeepFace", ["WEIGHTS_URL"]),
    ("facial_recognition", "ArcFace", ["WEIGHTS_URL"]),
    ("facial_recognition", "DeepID", ["WEIGHTS_URL"]),
    ("facial_recognition", "SFace", ["WEIGHTS_URL"]),
    ("facial_recognition", "GhostFaceNet", ["WEIGHTS_URL"]),
    ("facial_recognition", "Dlib", ["WEIGHT_URL"]),
    # demography
    ("facial_attribute", "Age", ["WEIGHTS_URL"]),
    ("facial_attribute", "Gender", ["WEIGHTS_URL"]),
    ("facial_attribute", "Race", ["WEIGHTS_URL"]),
    ("facial_attribute", "Emotion", ["WEIGHTS_URL"]),
    # spoofing
    ("spoofing", "Fasnet", ["FIRST_WEIGHTS_URL", "SECOND_WEIGHTS_URL"]),
    # face detection
    ("face_detector", "ssd", ["MODEL_URL", "WEIGHTS_URL"]),
    ("face_detector", "yunet", ["WEIGHTS_URL"]),
    ("face_detector", "dlib", ["WEIGHTS_URL"]),
    ("face_detector", "centerface", ["WEIGHTS_URL"]),
]

# models whose weight file is not named after its url
WEIGHT_FILE_NAMES = {
    ("facial_recognition", "GhostFaceNet"): {
        "tensorflow": "ghostfacenet_v1.h5",
        "pytorch": "ghostfacenet_v1.pth",
    },
}


def download_all_models_in_one_shot() -> None:
    """
    Download all model weights in one shot. Models that are not implemented for the
    backend engine in use are skipped.
    """

    # import here to avoid circular import issue
    import importlib
    from deepface.commons import backend_utils
    from deepface.modules import modeling
    from deepface.models.face_detection.Yolo import YoloModel

    backend = backend_utils.get_backend_engine()

    WEIGHTS: List[Union[str, Dict[str, str]]] = []

    for task, model_name, attributes in WEIGHT_SOURCES:
        try:
            model_class = modeling.get_model_class(task=task, model_name=model_name)
        except UnimplementedError:
            logger.info(f"{task}/{model_name} is not available for {backend}, skipping it")
            continue

        module = importlib.import_module(model_class.__module__)
        file_name = WEIGHT_FILE_NAMES.get((task, model_name), {}).get(backend)

        for attribute in attributes:
            url = getattr(module, attribute)
            WEIGHTS.append({"filename": file_name, "url": url} if file_name else url)

    # yolo detectors keep their file names next to their urls
    for model in YoloModel:
        WEIGHTS.append({"filename": model.value[0], "url": model.value[1]})

    for i in WEIGHTS:
        if isinstance(i, str):
            url = i
            filename = i.split("/")[-1]
            compress_type = None
            # if compressed file will be downloaded, get rid of its extension
            if filename.endswith(tuple(ALLOWED_COMPRESS_TYPES)):
                for ext in ALLOWED_COMPRESS_TYPES:
                    compress_type = ext
                    if filename.endswith(f".{ext}"):
                        filename = filename[: -(len(ext) + 1)]
                        break
        elif isinstance(i, dict):
            filename = i["filename"]
            url = i["url"]
        else:
            raise UnimplementedError("unimplemented scenario")
        logger.info(
            f"Downloading {url} to ~/.deepface/weights/{filename} with {compress_type} compression"
        )
        download_weights_if_necessary(
            file_name=filename, source_url=url, compress_type=compress_type
        )
