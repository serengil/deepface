import io
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import requests

from deepface.commons import image_utils


@pytest.fixture
def image_response(monkeypatch):
    pixels = np.array([[[12, 34, 56], [255, 0, 0]]], dtype=np.uint8)
    encoded, contents = cv2.imencode(".png", pixels)
    assert encoded
    response = requests.Response()
    response.status_code = 200
    response.url = "https://example.com/image.png"
    response.raw = io.BytesIO(contents.tobytes())
    get = Mock(return_value=response)
    monkeypatch.setattr(image_utils.requests, "get", get)
    return response, get, pixels


def test_http_image_response_is_closed_after_decoding(image_response):
    response, get, pixels = image_response

    image, name = image_utils.load_image(response.url)

    np.testing.assert_array_equal(image, pixels)
    assert name == response.url
    get.assert_called_once_with(response.url, stream=True, timeout=60)
    assert response.raw.closed


def test_http_image_response_is_closed_after_http_error(image_response):
    response, _, _ = image_response
    response.status_code = 503

    with pytest.raises(requests.HTTPError, match="503"):
        image_utils.load_image_from_web(response.url)

    assert response.raw.closed


def test_http_image_response_is_closed_after_read_error(image_response):
    response, _, _ = image_response
    failure = OSError("interrupted response body")

    class FailingRead(io.BytesIO):
        def read(self, *args, **kwargs):
            raise failure

    response.raw.close()
    response.raw = FailingRead()

    with pytest.raises(OSError) as exc:
        image_utils.load_image_from_web(response.url)

    assert exc.value is failure
    assert response.raw.closed


def test_http_image_response_is_closed_after_decode_error(image_response, monkeypatch):
    response, _, _ = image_response
    failure = ValueError("decoder failed")
    monkeypatch.setattr(image_utils.cv2, "imdecode", Mock(side_effect=failure))

    with pytest.raises(ValueError) as exc:
        image_utils.load_image_from_web(response.url)

    assert exc.value is failure
    assert response.raw.closed
