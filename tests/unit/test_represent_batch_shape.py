import numpy as np
import pytest

from deepface import DeepFace


@pytest.mark.parametrize("batch_size", [0, 1, 2])
@pytest.mark.parametrize("array_input", [False, True])
def test_representation_preserves_batch_structure(batch_size, array_input):
    images = np.random.default_rng(7).integers(
        0, 256, size=(batch_size, 24, 40, 3), dtype=np.uint8
    )
    source = images if array_input else list(images)
    options = {"model_name": "SFace", "detector_backend": "skip"}

    results = DeepFace.represent(source, **options)

    assert len(results) == batch_size
    for image, faces in zip(images, results):
        assert isinstance(faces, list)
        assert len(faces) == 1
        expected = DeepFace.represent(image, **options)
        np.testing.assert_allclose(faces[0]["embedding"], expected[0]["embedding"])
        assert faces[0]["facial_area"] == expected[0]["facial_area"]


def test_single_image_representation_keeps_flat_results():
    image = np.full((24, 40, 3), [20, 100, 220], dtype=np.uint8)

    results = DeepFace.represent(image, model_name="SFace", detector_backend="skip")

    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert len(results[0]["embedding"]) == 128
    assert np.isfinite(results[0]["embedding"]).all()
