import base64
import builtins
import gzip
import io
import os
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import requests
from urllib3.response import HTTPResponse

from deepface.commons import image_utils


@pytest.fixture
def png():
    image = np.array([[[81, 42, 20], [250, 10, 0]]], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return image, encoded.tobytes()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX named pipe")
def test_scan_skips_named_pipes(tmp_path, monkeypatch, png):
    photo = tmp_path / "photo.png"
    photo.write_bytes(png[1])
    fifo = tmp_path / "pipe.jpg"
    os.mkfifo(fifo)
    original_open = builtins.open

    def never_block(path, *args, **kwargs):
        assert os.fspath(path) != str(fifo), "tried to open a FIFO"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(image_utils, "open", never_block, raising=False)
    assert image_utils.list_images(str(tmp_path)) == [str(photo)]
    assert list(image_utils.yield_images(str(tmp_path))) == [str(photo)]


def test_scan_tolerates_removal_between_walk_and_open(tmp_path, monkeypatch, png):
    retained = tmp_path / "still-here.png"
    vanished = tmp_path / "gone.jpg"
    retained.write_bytes(png[1])
    vanished.write_bytes(png[1])
    original_open = builtins.open

    def remove_then_open(path, *args, **kwargs):
        if os.fspath(path) == str(vanished):
            vanished.unlink()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(image_utils, "open", remove_then_open, raising=False)
    assert image_utils.list_images(str(tmp_path)) == [str(retained)]


def test_header_file_is_closed_before_yield(tmp_path, monkeypatch, png):
    photo = tmp_path / "photo.png"
    photo.write_bytes(png[1])
    header = io.BytesIO(png[1])
    monkeypatch.setattr(image_utils, "open", Mock(return_value=header), raising=False)
    images = image_utils.yield_images(str(tmp_path))
    assert next(images) == str(photo)
    assert header.closed


def test_http_content_encoding_is_decoded_before_opencv(monkeypatch, png):
    response = requests.Response()
    response.status_code = 200
    response.raw = HTTPResponse(
        body=io.BytesIO(gzip.compress(png[1])),
        headers={"Content-Encoding": "gzip"},
        preload_content=False,
        decode_content=False,
    )
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    image = image_utils.load_image_from_web("https://example.com/photo.png")
    np.testing.assert_array_equal(image, png[0])


def test_corrupt_input_raises_at_the_decoder(tmp_path, monkeypatch):
    truncated = image_utils.PNG_MAGIC + b"broken"
    path = tmp_path / "broken.png"
    path.write_bytes(truncated)
    uri = "data:image/png;base64," + base64.b64encode(truncated).decode("ascii")
    response = requests.Response()
    response.status_code = 200
    response.raw = io.BytesIO(truncated)
    monkeypatch.setattr(requests, "get", Mock(return_value=response))

    for source in (str(path), uri, "https://example.com/broken.png", io.BytesIO(truncated)):
        with pytest.raises(ValueError, match="decode image"):
            image_utils.load_image(source)


def test_empty_stream_raises_decode_error():
    with pytest.raises(ValueError, match="decode image"):
        image_utils.load_image(io.BytesIO())
