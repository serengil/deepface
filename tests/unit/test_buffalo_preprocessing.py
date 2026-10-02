# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.models.facial_recognition.onnx.Buffalo_L import Buffalo_L
from deepface.modules import preprocessing


@pytest.mark.parametrize("batch_size", [1, 2])
def test_buffalo_preprocessing_matches_insightface_input(batch_size):
    # Different values in each BGR channel expose an accidental channel reversal.
    face = np.empty((112, 112, 3), dtype=np.uint8)
    face[:] = [32, 96, 224]
    normalized = preprocessing.resize_image(face, (112, 112))
    expected = np.expand_dims(face, axis=0)
    if batch_size == 2:
        normalized = np.concatenate([normalized, normalized / 2], axis=0)
        expected = np.concatenate([expected, expected / 2], axis=0)
    original = normalized.copy()
    client = Buffalo_L.__new__(Buffalo_L)

    result = client.preprocess(normalized)

    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(normalized, original)


def test_buffalo_single_image_keeps_batch_dimension():
    client = Buffalo_L.__new__(Buffalo_L)
    face = np.zeros((112, 112, 3), dtype=np.float32)
    assert client.preprocess(face).shape == (1, 112, 112, 3)
