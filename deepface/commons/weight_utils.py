# built-in dependencies
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union
import zipfile
import bz2

# 3rd party dependencies
import gdown
import requests  # type: ignore[import-untyped]

# project dependencies
from deepface.commons import folder_utils
from deepface.commons.logger import Logger
from deepface.modules.exceptions import UnimplementedError

logger = Logger()

# pylint: disable=line-too-long, use-maxsplit-arg

ALLOWED_COMPRESS_TYPES = ["zip", "bz2"]


def download_weights_if_necessary(
    file_name: str, source_url: Union[str, List[str]], compress_type: Optional[str] = None
) -> str:
    """
    Download the weights of a pre-trained model from external source if not downloaded yet.
    Args:
        file_name (str): target file name with extension
        source_url (str or list of str): source url to be downloaded. if a list is given,
            the urls are tried in order and the next one is used as a backup when
            downloading from the previous one fails
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

    source_urls = [source_url] if isinstance(source_url, str) else list(source_url)
    if len(source_urls) == 0:
        raise ValueError(f"no source url is given to download {file_name}")

    last_err: Optional[Exception] = None
    for idx, url in enumerate(source_urls):
        try:
            logger.info(f"🔗 {file_name} will be downloaded from {url} to {target_file}...")
            ensure_source_is_reachable(url)
            if compress_type is None:
                gdown.download(url, target_file, quiet=False)
            else:
                # Stage each attempt on the cache filesystem. Readers should only
                # see a complete weight, including when a mirror serves a bad archive.
                with tempfile.TemporaryDirectory(dir=os.path.dirname(target_file)) as staging:
                    archive = os.path.join(staging, "download")
                    extracted = os.path.join(staging, "weight")
                    gdown.download(url, archive, quiet=False)
                    with open(extracted, "wb") as dest:
                        if compress_type == "zip":
                            with zipfile.ZipFile(archive) as zipped, zipped.open(file_name) as src:
                                shutil.copyfileobj(src, dest)
                        else:
                            with bz2.open(archive, "rb") as compressed:
                                shutil.copyfileobj(compressed, dest)
                    os.replace(extracted, target_file)
            last_err = None
            break
        except Exception as err:  # pylint: disable=broad-except
            last_err = err
            # do not let a partially downloaded file be used by the next attempt
            if compress_type is None and os.path.isfile(target_file):
                os.remove(target_file)
            if idx < len(source_urls) - 1:
                logger.warn(
                    f"Downloading {file_name} from {url} failed ({err}). "
                    f"Trying the backup source {source_urls[idx + 1]}..."
                )

    if last_err is not None:
        raise ValueError(
            f"⛓️‍💥 An exception occurred while downloading {file_name} from "
            f"{', '.join(source_urls)}. "
            f"Consider downloading it manually to {target_file}."
        ) from last_err

    return target_file


def ensure_source_is_reachable(url: str) -> None:
    """
    gdown does not raise for http errors of non google drive urls, and saves the error
    page (e.g. "Not Found") as if it were the weight file. check the status in advance
    to fail fast, so that the backup source can be tried.
    Args:
        url (str): source url to be downloaded
    """
    response = requests.head(url, allow_redirects=True, timeout=30)
    # some servers do not allow head requests, let gdown decide for them
    if response.status_code >= 400 and response.status_code != 405:
        raise ValueError(f"{url} responded with status code {response.status_code}")


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
# implemented for the backend engine in use, or when its implementation for that engine
# does not download any weights. the file name is the last part of the url unless it is
# overwritten here
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
    # mtcnn downloads weights on onnx only, its tensorflow and pytorch packages bring theirs
    ("face_detector", "mtcnn", ["PNET_WEIGHTS_URL", "RNET_WEIGHTS_URL", "ONET_WEIGHTS_URL"]),
]

# models whose weight file is not named after its url
WEIGHT_FILE_NAMES = {
    # its tensorflow weights are served by their author, under a different name
    ("facial_recognition", "GhostFaceNet"): {
        "tensorflow": "ghostfacenet_v1.h5",
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
    from deepface.models.face_detection.pytorch.Yolo import YoloModel

    backend = backend_utils.get_backend_engine()

    WEIGHTS: List[Union[str, List[str], Dict[str, Any]]] = []

    for task, model_name, attributes in WEIGHT_SOURCES:
        try:
            model_class = modeling.get_model_class(task=task, model_name=model_name)
        except UnimplementedError:
            logger.info(f"{task}/{model_name} is not available for {backend}, skipping it")
            continue

        module = importlib.import_module(model_class.__module__)
        file_name = WEIGHT_FILE_NAMES.get((task, model_name), {}).get(backend)

        for attribute in attributes:
            url = getattr(module, attribute, None)
            if url is None:
                logger.info(f"{task}/{model_name} downloads no weights on {backend}, skipping it")
                continue
            WEIGHTS.append({"filename": file_name, "url": url} if file_name else url)

    # yolo detectors keep their file names next to their urls
    for model in YoloModel:
        WEIGHTS.append({"filename": model.value[0], "url": model.value[1]})

    for i in WEIGHTS:
        compress_type = None
        if isinstance(i, (str, list)):
            url = i
            # backup sources serve the same file, so the name comes from the primary one
            primary_url = i if isinstance(i, str) else i[0]
            filename = primary_url.split("/")[-1]
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
