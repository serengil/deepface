# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.models.facial_recognition.onnx.Buffalo_L import Buffalo_L
from deepface.modules import preprocessing


def insightface_blob(faces):
    # cv2.dnn.blobFromImages(faces, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    # as insightface's ArcFaceONNX.get_feat calls it
    rgb = np.stack(faces).astype(np.float32)[..., ::-1]
    return ((rgb - 127.5) / 127.5).transpose(0, 3, 1, 2)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_buffalo_preprocessing_matches_insightface_input(batch_size):
    # Different values in each BGR channel expose an accidental channel reversal.
    face = np.empty((112, 112, 3), dtype=np.uint8)
    face[:] = [32, 96, 224]
    normalized = preprocessing.resize_image(face, (112, 112))
    faces = [face]
    if batch_size == 2:
        normalized = np.concatenate([normalized, normalized / 2], axis=0)
        faces.append(face / 2)
    original = normalized.copy()
    client = Buffalo_L.__new__(Buffalo_L)

    result = client.preprocess(normalized)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, insightface_blob(faces), rtol=0, atol=1e-5)
    np.testing.assert_array_equal(normalized, original)


def test_buffalo_single_image_keeps_batch_dimension():
    client = Buffalo_L.__new__(Buffalo_L)
    face = np.zeros((112, 112, 3), dtype=np.float32)
    assert client.preprocess(face).shape == (1, 3, 112, 112)
