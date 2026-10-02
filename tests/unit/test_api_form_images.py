import pytest
from flask import Flask

from deepface.api.src.app import create_app
from deepface.api.src.modules.core import routes


@pytest.mark.parametrize(
    "content_type", ["application/x-www-form-urlencoded", "multipart/form-data"]
)
@pytest.mark.parametrize(
    "image",
    ["image.jpg", "https://example.com/image.jpg", "data:image/jpeg;base64,aW1n"],
)
def test_extract_string_image_from_form(content_type, image):
    app = Flask(__name__)
    with app.test_request_context(
        "/represent", method="POST", data={"img": image}, content_type=content_type
    ):
        assert routes.extract_image_from_request("img") == image


def test_extract_string_image_from_json():
    app = Flask(__name__)
    with app.test_request_context(
        "/represent", method="POST", json={"img": "image.jpg"}
    ):
        assert routes.extract_image_from_request("img") == "image.jpg"


@pytest.mark.parametrize(
    "content_type", ["application/x-www-form-urlencoded", "multipart/form-data"]
)
def test_represent_accepts_form_image(monkeypatch, content_type):
    monkeypatch.delenv("DEEPFACE_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        routes.service,
        "represent",
        lambda **kwargs: ({"image": kwargs["img_path"]}, 200),
    )
    client = create_app().test_client()
    response = client.post(
        "/represent", data={"img": "image.jpg"}, content_type=content_type
    )
    assert response.status_code == 200
    assert response.json == {"image": "image.jpg"}


@pytest.mark.parametrize(
    "content_type", ["application/x-www-form-urlencoded", "multipart/form-data"]
)
def test_missing_image_in_form(content_type):
    app = Flask(__name__)
    with app.test_request_context(
        "/represent",
        method="POST",
        data={"model_name": "Facenet"},
        content_type=content_type,
    ):
        with pytest.raises(ValueError, match="'img' not found"):
            routes.extract_image_from_request("img")
