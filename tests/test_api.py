# built-in dependencies
import base64
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

# 3rd party dependencies
import flask
import gdown
import numpy as np
import werkzeug
from flask import Flask
from packaging import version

# project dependencies
from deepface.api.src.app import create_app
from deepface.api.src.modules.core import routes
from deepface.commons.logger import Logger
from deepface.modules.detection import extract_faces, DetectedFace, FacialAreaRegion
from deepface.modules import detection

logger = Logger()

IMG1_SOURCE = (
    "https://raw.githubusercontent.com/serengil/deepface/refs/heads/master/tests/dataset/img1.jpg"
)
IMG2_SOURCE = (
    "https://raw.githubusercontent.com/serengil/deepface/refs/heads/master/tests/dataset/img2.jpg"
)
DUMMY_APP = Flask(__name__)


class TestVerifyEndpoint(unittest.TestCase):
    def setUp(self):
        download_test_images(IMG1_SOURCE)
        download_test_images(IMG2_SOURCE)
        app = create_app()
        app.config["DEBUG"] = True
        app.config["TESTING"] = True
        self.app = app.test_client()

    def setup_invalid_image(self):
        """Create a file with invalid image data in a temporary directory."""
        self.tempdir = TemporaryDirectory(prefix="test-", dir=".")
        self.addCleanup(shutil.rmtree, self.tempdir.name)
        self.invalid_image_path = Path(self.tempdir.name) / "invalid.jpg"
        self.invalid_image_path.write_text("Not JPEG data")

    def test_tp_verify(self):
        data = {
            "img1": "dataset/img1.jpg",
            "img2": "dataset/img2.jpg",
        }
        response = self.app.post("/verify", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)

        assert result.get("verified") is not None
        assert result.get("model") is not None
        assert result.get("similarity_metric") is not None
        assert result.get("detector_backend") is not None
        assert result.get("distance") is not None
        assert result.get("threshold") is not None
        assert result.get("facial_areas") is not None

        assert result.get("verified") is True

        logger.info("✅ true-positive verification api test is done")

    def test_tn_verify(self):
        data = {
            "img1": "dataset/img1.jpg",
            "img2": "dataset/img2.jpg",
        }
        response = self.app.post("/verify", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)

        assert result.get("verified") is not None
        assert result.get("model") is not None
        assert result.get("similarity_metric") is not None
        assert result.get("detector_backend") is not None
        assert result.get("distance") is not None
        assert result.get("threshold") is not None
        assert result.get("facial_areas") is not None

        assert result.get("verified") is True

        logger.info("✅ true-negative verification api test is done")

    def test_represent(self):
        data = {
            "img": "dataset/img1.jpg",
        }
        response = self.app.post("/represent", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)
        assert result.get("results") is not None
        assert isinstance(result["results"], list) is True
        assert len(result["results"]) > 0
        for i in result["results"]:
            assert i.get("embedding") is not None
            assert isinstance(i.get("embedding"), list) is True
            assert len(i.get("embedding")) == 4096
            assert i.get("face_confidence") is not None
            assert i.get("facial_area") is not None

        logger.info("✅ representation api test is done (for image path)")

    def test_represent_encoded(self):
        image_path = "dataset/img1.jpg"
        with open(image_path, "rb") as image_file:
            encoded_string = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode(
                "utf8"
            )

        data = {"model_name": "Facenet", "detector_backend": "mtcnn", "img": encoded_string}

        response = self.app.post("/represent", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)
        assert result.get("results") is not None
        assert isinstance(result["results"], list) is True
        assert len(result["results"]) > 0
        for i in result["results"]:
            assert i.get("embedding") is not None
            assert isinstance(i.get("embedding"), list) is True
            assert len(i.get("embedding")) == 128
            assert i.get("face_confidence") is not None
            assert i.get("facial_area") is not None

        logger.info("✅ representation api test is done (for encoded image)")

    def test_represent_url(self):
        data = {
            "model_name": "Facenet",
            "detector_backend": "mtcnn",
            "img": "https://github.com/serengil/deepface/blob/master/tests/dataset/couple.jpg?raw=true",
        }

        response = self.app.post("/represent", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)
        assert result.get("results") is not None
        assert isinstance(result["results"], list) is True
        assert len(result["results"]) == 2  # 2 faces are in the image link
        for i in result["results"]:
            assert i.get("embedding") is not None
            assert isinstance(i.get("embedding"), list) is True
            assert len(i.get("embedding")) == 128
            assert i.get("face_confidence") is not None
            assert i.get("facial_area") is not None

        logger.info("✅ representation api test is done (for image url)")

    def test_analyze(self):
        data = {
            "img": "dataset/img1.jpg",
        }
        response = self.app.post("/analyze", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)
        assert result.get("results") is not None
        assert isinstance(result["results"], list) is True
        assert len(result["results"]) > 0
        for i in result["results"]:
            assert i.get("age") is not None
            assert isinstance(i.get("age"), (int, float))
            assert i.get("dominant_gender") is not None
            assert i.get("dominant_gender") in ["Man", "Woman"]
            assert i.get("dominant_emotion") is not None
            assert i.get("dominant_race") is not None

        logger.info("✅ analyze api test is done")

    def test_analyze_inputformats(self):
        image_path = "dataset/couple.jpg"
        with open(image_path, "rb") as image_file:
            encoded_image = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode(
                "utf8"
            )

        image_sources = [
            # image path
            image_path,
            # image url
            f"https://github.com/serengil/deepface/blob/master/tests/{image_path}?raw=true",
            # encoded image
            encoded_image,
        ]

        results = []
        for img in image_sources:
            data = {
                "img": img,
            }
            response = self.app.post("/analyze", json=data)

            assert response.status_code == 200
            result = response.json
            results.append(result)

            assert result.get("results") is not None
            assert isinstance(result["results"], list) is True
            assert len(result["results"]) > 0
            for i in result["results"]:
                assert i.get("age") is not None
                assert isinstance(i.get("age"), (int, float))
                assert i.get("dominant_gender") is not None
                assert i.get("dominant_gender") in ["Man", "Woman"]
                assert i.get("dominant_emotion") is not None
                assert i.get("dominant_race") is not None

        assert len(results[0]["results"]) == len(results[1]["results"]) and len(
            results[0]["results"]
        ) == len(results[2]["results"])

        for i in range(len(results[0]["results"])):
            assert (
                results[0]["results"][i]["dominant_emotion"]
                == results[1]["results"][i]["dominant_emotion"]
                and results[0]["results"][i]["dominant_emotion"]
                == results[2]["results"][i]["dominant_emotion"]
            )

            assert (
                results[0]["results"][i]["dominant_gender"]
                == results[1]["results"][i]["dominant_gender"]
                and results[0]["results"][i]["dominant_gender"]
                == results[2]["results"][i]["dominant_gender"]
            )

            assert (
                results[0]["results"][i]["dominant_race"]
                == results[1]["results"][i]["dominant_race"]
                and results[0]["results"][i]["dominant_race"]
                == results[2]["results"][i]["dominant_race"]
            )

        logger.info("✅ different inputs test is done")

    def test_invalid_verify(self):
        data = {
            "img1": "dataset/invalid_1.jpg",
            "img2": "dataset/invalid_2.jpg",
        }
        response = self.app.post("/verify", json=data)
        assert response.status_code == 400
        logger.info("✅ invalid verification request api test is done")

    def test_invalid_represent(self):
        """/represent fails with status 400  for invalid image data"""
        self.setup_invalid_image()
        data = {
            "img": str(self.invalid_image_path),
        }
        response = self.app.post("/represent", json=data)
        assert response.status_code == 400
        logger.info("✅ invalid represent request api test is done")

    def test_no_file_represent(self):
        """/represent fails with status 400  for missing image file"""
        data = {
            "img": "no_such_file.jpg",
        }
        response = self.app.post("/represent", json=data)
        assert response.status_code == 400

    def test_invalid_analyze(self):
        """/analyze fails with status 400  for invalid image data"""
        self.setup_invalid_image()
        data = {
            "img": str(self.invalid_image_path),
        }
        response = self.app.post("/analyze", json=data)
        assert response.status_code == 400

    def test_no_file_analyze(self):
        """/analyze fails with status 400  for missing image file"""
        data = {
            "img": "no_such_file.jpg",
        }
        response = self.app.post("/analyze", json=data)
        assert response.status_code == 400

    def test_analyze_for_multipart_form_data(self):
        if is_form_data_file_testable() is False:
            return

        with open("/tmp/img1.jpg", "rb") as img_file:
            response = self.app.post(
                "/analyze",
                content_type="multipart/form-data",
                data={
                    "img": (img_file, "test_image.jpg"),
                    "actions": '["age", "gender"]',
                    "detector_backend": "mtcnn",
                },
            )
            assert response.status_code == 200
            result = response.json
            assert isinstance(result, dict)
            assert result.get("age") is not True
            assert result.get("dominant_gender") is not True
            logger.info("✅ analyze api for multipart form data test is done")

    def test_verify_for_multipart_form_data(self):
        if is_form_data_file_testable() is False:
            return

        with open("/tmp/img1.jpg", "rb") as img1_file:
            with open("/tmp/img2.jpg", "rb") as img2_file:
                response = self.app.post(
                    "/verify",
                    content_type="multipart/form-data",
                    data={
                        "img1": (img1_file, "first_image.jpg"),
                        "img2": (img2_file, "second_image.jpg"),
                        "model_name": "Facenet",
                        "detector_backend": "mtcnn",
                        "distance_metric": "euclidean",
                    },
                )
                assert response.status_code == 200
                result = response.json
                assert isinstance(result, dict)
                assert result.get("verified") is not None
                assert result.get("model") == "Facenet"
                assert result.get("similarity_metric") is not None
                assert result.get("detector_backend") == "mtcnn"
                assert result.get("threshold") is not None
                assert result.get("facial_areas") is not None

                logger.info("✅ verify api for multipart form data test is done")

    def test_represent_for_multipart_form_data(self):
        if is_form_data_file_testable() is False:
            return

        with open("/tmp/img1.jpg", "rb") as img_file:
            response = self.app.post(
                "/represent",
                content_type="multipart/form-data",
                data={
                    "img": (img_file, "first_image.jpg"),
                    "model_name": "Facenet",
                    "detector_backend": "mtcnn",
                },
            )
            assert response.status_code == 200
            result = response.json
            assert isinstance(result, dict)
            logger.info("✅ represent api for multipart form data test is done")

    def test_represent_for_multipart_form_data_and_filepath(self):
        if is_form_data_file_testable() is False:
            return

        response = self.app.post(
            "/represent",
            content_type="multipart/form-data",
            data={
                "img": "/tmp/img1.jpg",
                "model_name": "Facenet",
                "detector_backend": "mtcnn",
            },
        )
        assert response.status_code == 200
        result = response.json
        assert isinstance(result, dict)
        logger.info("✅ represent api for multipart form data and file path test is done")

    def test_extract_image_from_form_data(self):
        if is_form_data_file_testable() is False:
            return

        img_key = "img1"
        img_itself = np.zeros((100, 100, 3), dtype=np.uint8)
        # Establish a temporary request context using the Flask app
        with DUMMY_APP.test_request_context("/dummy_endpoint"):
            # Mock the file part
            with patch("deepface.api.src.modules.core.routes.request") as mock_request:
                mock_file = MagicMock()
                mock_file.filename = "image.jpg"
                mock_request.files = {img_key: mock_file}

                # Mock the image loading function
                with patch(
                    "deepface.commons.image_utils.load_image_from_file_storage",
                    return_value=img_itself,
                ):
                    result = routes.extract_image_from_request(img_key)

                    assert isinstance(result, np.ndarray)
                    assert np.array_equal(result, img_itself)

        logger.info("✅ test extract_image_from_request for real image from form data done")

    def test_extract_image_string_from_json_data(self):
        if is_form_data_file_testable() is False:
            return

        img_key = "img1"
        img_data = "image_url_or_path_or_base64"

        with DUMMY_APP.test_request_context("/dummy_endpoint"):
            with patch("deepface.api.src.modules.core.routes.request") as mock_request:
                # Mock JSON data
                mock_request.files = None
                mock_request.is_json = True
                mock_request.get_json = MagicMock(return_value={img_key: img_data})

                result = routes.extract_image_from_request(img_key)

                assert isinstance(result, str)
                assert result == img_data

        logger.info("✅ test extract_image_from_request for image string from json done")

    def test_extract_image_string_from_form_data(self):
        if is_form_data_file_testable() is False:
            return

        img_key = "img1"
        img_data = "image_url_or_path_or_base64"

        with DUMMY_APP.test_request_context("/dummy_endpoint"):
            with patch("deepface.api.src.modules.core.routes.request") as mock_request:
                # Mock form data
                mock_request.files = None

                mock_request.is_json = False
                mock_request.get_json = MagicMock(return_value=None)

                mock_request.form = MagicMock()
                mock_request.form.to_dict.return_value = {img_key: img_data}

                result = routes.extract_image_from_request(img_key)

                assert isinstance(result, str)
                assert result == img_data

        logger.info("✅ test extract_image_from_request for image string from form done")


def download_test_images(url: str):
    file_name = url.split("/")[-1]
    target_file = f"/tmp/{file_name}"
    if os.path.exists(target_file) is True:
        return

    gdown.download(url, target_file, quiet=False)


def is_form_data_file_testable() -> bool:
    """
    Sending a file from form data fails in unit test with
        415 unsupported media type error for flask 3.X
        but it is working for flask 2.0.2
    Returns:
        is_form_data_file_testable (bool)
    """
    flask_version = version.parse(flask.__version__)
    werkzeus_version = version.parse(werkzeug.__version__)
    threshold_version = version.parse("2.0.2")
    is_testable = flask_version <= threshold_version and werkzeus_version <= threshold_version
    if is_testable is False:
        logger.warn(
            "sending file in form data is not testable because of flask, werkzeus versions."
            f"Expected <= {threshold_version}, but {flask_version=} and {werkzeus_version}."
        )
    return is_testable


def test_landmarks_are_raw_python_types(monkeypatch):
    """
    Tests that the landmarks returned by extract_faces are of raw Python types (int, tuple of ints).
        This is important for compatibility with frameworks like Flask that may have issues
        serializing NumPy types.
    """
    fake_landmarks = {
        "x": 10,
        "y": np.int64(15),
        "w": 200,
        "h": 200,
        "left_eye": np.array([50, 60], dtype=np.int64),
        "right_eye": np.array([150, 60], dtype=np.int32),
        "nose": (100, 100),
        "confidence": 1,
    }

    def fake_detect_faces(**kwargs):
        return [
            DetectedFace(
                img=np.zeros((200, 200, 3), dtype=np.uint8),
                facial_area=FacialAreaRegion(
                    x=fake_landmarks["x"],
                    y=fake_landmarks["y"],
                    w=fake_landmarks["w"],
                    h=fake_landmarks["h"],
                    confidence=fake_landmarks["confidence"],
                    left_eye=fake_landmarks["left_eye"],
                    right_eye=fake_landmarks["right_eye"],
                    nose=fake_landmarks["nose"],
                ),
                confidence=fake_landmarks["confidence"],
            )
        ]

    monkeypatch.setattr(detection, "detect_faces", fake_detect_faces)

    def fake_load_image(*args, **kwargs):
        return np.zeros((200, 200, 3), dtype=np.uint8), "fake.jpg"

    monkeypatch.setattr(detection.image_utils, "load_image", fake_load_image)

    results = extract_faces("fake_path.jpg", detector_backend="opencv", enforce_detection=True)

    landmarks = results[0]["facial_area"]
    for _, value in landmarks.items():
        if isinstance(value, tuple):
            assert all(isinstance(coord, int) for coord in value)
        else:
            assert isinstance(value, int)
