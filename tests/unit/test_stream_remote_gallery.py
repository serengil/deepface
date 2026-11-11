# built-in dependencies
import io
import sys
from types import SimpleNamespace
from unittest.mock import Mock

# 3rd party dependencies
import cv2
import numpy as np
import pytest

# project dependencies
from deepface.modules import streaming


@pytest.mark.parametrize("face_count", [1, 2])
def test_stream_displays_s3_candidate_with_real_image_decode(monkeypatch, face_count):
    pixels = np.full((12, 9, 3), (10, 20, 30), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", pixels)
    assert success
    body = io.BytesIO(encoded.tobytes())
    client = Mock()
    client.get_object.return_value = {"Body": body}
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *args: client))
    monkeypatch.setattr(streaming.DeepFace, "find", Mock(return_value=[[
        {"identity": "s3://bucket/gallery/person.png", "confidence": 98.0}
    ]]))

    def extract(img_path, **kwargs):
        # Use the same decoder as extract_faces; an s3:// string is not an image input.
        from deepface.commons import image_utils
        image, _ = image_utils.load_image(img_path)
        np.testing.assert_array_equal(image, pixels)
        return [{"face": image[:, :, ::-1].astype(float) / 255} for _ in range(face_count)]

    monkeypatch.setattr(streaming.DeepFace, "extract_faces", extract)
    name, image, confidence = streaming.search_identity(
        pixels, "s3://bucket/gallery", "VGG-Face", "opencv", "cosine"
    )
    assert name == "person.png"
    assert confidence == 98.0
    assert image.shape == (112, 112, 3)
    np.testing.assert_allclose(image[0, 0], [10, 20, 30])
    client.get_object.assert_called_once_with(Bucket="bucket", Key="gallery/person.png")
    assert body.closed
