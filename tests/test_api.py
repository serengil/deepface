import unittest
from deepface.commons.logger import Logger
from deepface.api.src.app import create_app
import base64


logger = Logger("tests/test_api.py")


class TestVerifyEndpoint(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.config["DEBUG"] = True
        app.config["TESTING"] = True
        self.app = app.test_client()

    def test_tp_verify(self):
        data = {
            "img1_path": "dataset/img1.jpg",
            "img2_path": "dataset/img2.jpg",
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
            "img1_path": "dataset/img1.jpg",
            "img2_path": "dataset/img2.jpg",
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
            encoded_string = "data:image/jpeg;base64," + \
                base64.b64encode(image_file.read()).decode("utf8")

        data = {
            "model_name": "Facenet",
            "detector_backend": "mtcnn",
            "img": encoded_string
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
            assert len(i.get("embedding")) == 128
            assert i.get("face_confidence") is not None
            assert i.get("facial_area") is not None

        logger.info("✅ representation api test is done (for encoded image)")

    def test_represent_url(self):
        data = {
            "model_name": "Facenet",
            "detector_backend": "mtcnn",
            "img": "https://github.com/serengil/deepface/blob/master/tests/dataset/couple.jpg?raw=true"
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
            encoded_image = "data:image/jpeg;base64," + \
                base64.b64encode(image_file.read()).decode("utf8")

        image_sources = [
            # image path
            image_path,
            # image url
            f"https://github.com/serengil/deepface/blob/master/tests/{image_path}?raw=true",
            # encoded image
            encoded_image
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

        assert len(results[0]["results"]) == len(results[1]["results"])\
            and len(results[0]["results"]) == len(results[2]["results"])

        for i in range(len(results[0]['results'])):
            assert results[0]["results"][i]["dominant_emotion"] == results[1]["results"][i]["dominant_emotion"]\
                and results[0]["results"][i]["dominant_emotion"] == results[2]["results"][i]["dominant_emotion"]

            assert results[0]["results"][i]["dominant_gender"] == results[1]["results"][i]["dominant_gender"]\
                and results[0]["results"][i]["dominant_gender"] == results[2]["results"][i]["dominant_gender"]

            assert results[0]["results"][i]["dominant_race"] == results[1]["results"][i]["dominant_race"]\
                and results[0]["results"][i]["dominant_race"] == results[2]["results"][i]["dominant_race"]

        logger.info("✅ different inputs test is done")

    def test_invalid_verify(self):
        data = {
            "img1_path": "dataset/invalid_1.jpg",
            "img2_path": "dataset/invalid_2.jpg",
        }
        response = self.app.post("/verify", json=data)
        assert response.status_code == 400
        logger.info("✅ invalid verification request api test is done")

    def test_invalid_represent(self):
        data = {
            "img": "dataset/invalid_1.jpg",
        }
        response = self.app.post("/represent", json=data)
        assert response.status_code == 400
        logger.info("✅ invalid represent request api test is done")

    def test_invalid_analyze(self):
        data = {
            "img": "dataset/invalid.jpg",
        }
        response = self.app.post("/analyze", json=data)
        assert response.status_code == 400
