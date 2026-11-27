import gzip
import io
import zlib
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import requests
from urllib3.response import HTTPResponse

from deepface.commons import image_utils


def response_for(body, encoding, extension=".png"):
    source = io.BytesIO(body)
    headers = {"Content-Type": "image/jpeg" if extension == ".jpg" else "image/png"}
    if encoding != "identity":
        headers["Content-Encoding"] = encoding
    response = requests.Response()
    response.status_code = 200
    response.url = f"https://example.com/image{extension}"
    response.headers.update(headers)
    response.raw = HTTPResponse(
        body=source, headers=headers, preload_content=False, decode_content=False
    )
    return response, source


@pytest.mark.parametrize("extension", [".png", ".jpg"])
@pytest.mark.parametrize("encoding", ["identity", "gzip", "deflate"])
def test_http_image_decodes_the_response_content(monkeypatch, extension, encoding):
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    encoded, contents = cv2.imencode(extension, pixels)
    assert encoded
    expected = cv2.imdecode(contents, cv2.IMREAD_COLOR)
    body = contents.tobytes()
    if encoding == "gzip":
        body = gzip.compress(body)
    elif encoding == "deflate":
        body = zlib.compress(body)
    response, source = response_for(body, encoding, extension)
    get = Mock(return_value=response)
    monkeypatch.setattr(image_utils.requests, "get", get)

    image, name = image_utils.load_image(response.url)

    assert image is not None
    np.testing.assert_array_equal(image, expected)
    assert name == response.url
    get.assert_called_once_with(response.url, stream=True, timeout=60)
    assert source.closed


def test_invalid_content_encoding_propagates_and_closes_the_response(monkeypatch):
    response, source = response_for(b"not a gzip stream", "gzip")
    monkeypatch.setattr(image_utils.requests, "get", Mock(return_value=response))

    with pytest.raises(requests.exceptions.ContentDecodingError):
        image_utils.load_image_from_web(response.url)

    assert source.closed
