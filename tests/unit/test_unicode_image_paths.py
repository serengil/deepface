from pathlib import Path

import cv2
import numpy as np
import pytest

from deepface.commons.image_utils import load_image


@pytest.mark.parametrize(
    "relative_path", ["照片/face.png", "faces/café.png", "顔/写真.png", "faces/face.png"]
)
@pytest.mark.parametrize("path_type", [str, Path])
def test_local_image_paths_preserve_pixels_and_names(tmp_path, relative_path, path_type):
    pixels = np.array([[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [12, 34, 56]]], dtype=np.uint8)
    encoded, contents = cv2.imencode(".png", pixels)
    assert encoded
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(contents.tobytes())

    image, name = load_image(path_type(path))

    np.testing.assert_array_equal(image, pixels)
    assert name == str(path)


@pytest.mark.parametrize("path_type", [str, Path])
def test_unicode_path_loads_when_opencv_filename_reader_raises(tmp_path, monkeypatch, path_type):
    pixels = np.array([[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [12, 34, 56]]], dtype=np.uint8)
    encoded, contents = cv2.imencode(".png", pixels)
    assert encoded
    path = tmp_path / "照片.png"
    path.write_bytes(contents.tobytes())

    def reject_filename(_filename):
        raise cv2.error("Unicode filenames are not supported")

    monkeypatch.setattr(cv2, "imread", reject_filename)

    image, name = load_image(path_type(path))

    np.testing.assert_array_equal(image, pixels)
    assert name == str(path)
