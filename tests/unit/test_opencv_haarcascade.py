# built-in dependencies
import os

# 3rd party dependencies
import pytest

# project dependencies
from deepface.models.face_detection import OpenCv
from deepface.commons.logger import Logger

logger = Logger()

BUNDLED_PATH = os.path.join(os.path.dirname(OpenCv.cv2.__file__), "data")


def test_haarcascade_files_are_downloaded_if_missing(monkeypatch, tmp_path):
    # simulate opencv 5 wheels, which do not ship haarcascade files
    monkeypatch.setattr(OpenCv.OpenCvClient, "_OpenCvClient__get_opencv_path", lambda self: str(tmp_path))

    downloads = []
    original_download = OpenCv.weight_utils.download_weights_if_necessary

    def download_weights_if_necessary(file_name, source_url):
        downloads.append((file_name, source_url))
        # bundled files are missing in opencv 5, so download them for real
        bundled_file = os.path.join(BUNDLED_PATH, file_name)
        if os.path.isfile(bundled_file):
            return bundled_file
        return original_download(file_name=file_name, source_url=source_url)

    monkeypatch.setattr(
        OpenCv.weight_utils, "download_weights_if_necessary", download_weights_if_necessary
    )

    client = OpenCv.OpenCvClient()

    assert downloads == [
        (
            "haarcascade_frontalface_default.xml",
            "https://github.com/opencv/opencv/raw/4.9.0/data/haarcascades/"
            "haarcascade_frontalface_default.xml",
        ),
        (
            "haarcascade_eye.xml",
            "https://github.com/opencv/opencv/raw/4.9.0/data/haarcascades/haarcascade_eye.xml",
        ),
    ]
    assert not client.model["face_detector"].empty()
    assert not client.model["eye_detector"].empty()
    logger.info("✅ test haarcascade files are downloaded if missing done")


def test_haarcascade_files_are_not_downloaded_if_bundled(monkeypatch):
    if not os.path.isfile(os.path.join(BUNDLED_PATH, "haarcascade_frontalface_default.xml")):
        pytest.skip(f"opencv {OpenCv.cv2.__version__} does not ship haarcascade files")

    def download_weights_if_necessary(**kwargs):
        raise AssertionError("bundled haarcascade files must be used")

    monkeypatch.setattr(
        OpenCv.weight_utils, "download_weights_if_necessary", download_weights_if_necessary
    )

    client = OpenCv.OpenCvClient()
    assert not client.model["face_detector"].empty()
    logger.info("✅ test haarcascade files are not downloaded if bundled done")


def test_cascade_classifier_is_required(monkeypatch):
    # simulate opencv 5 main package, which does not have cascade classifier
    monkeypatch.delattr(OpenCv.cv2, "CascadeClassifier")

    def download_weights_if_necessary(**kwargs):
        raise AssertionError("weights must not be downloaded without cascade classifier")

    monkeypatch.setattr(
        OpenCv.weight_utils, "download_weights_if_necessary", download_weights_if_necessary
    )

    with pytest.raises(ValueError, match="opencv-contrib-python-headless"):
        OpenCv.OpenCvClient()
    logger.info("✅ test cascade classifier is required done")
