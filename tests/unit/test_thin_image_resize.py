import numpy as np
import pytest

from deepface import DeepFace
from deepface.modules import preprocessing


@pytest.mark.parametrize("shape", [(1, 1000, 3), (1000, 1, 3)])
@pytest.mark.parametrize("target_size", [(112, 112), (47, 55)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_thin_images_keep_visible_pixels(shape, target_size, dtype):
    image = np.full(shape, [20, 100, 220], dtype=dtype)

    result = preprocessing.resize_image(image, target_size)

    assert result.shape == (1, *target_size, 3)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    visible = result[0, :, :, 2] > 0
    assert visible.any()
    assert (~visible).any()
    np.testing.assert_allclose(
        result[0][visible],
        np.broadcast_to(np.array([20, 100, 220]) / 255, result[0][visible].shape),
    )


@pytest.mark.parametrize("shape", [(1, 1, 3), (50, 100, 3), (100, 50, 3)])
def test_regular_image_resize_preserves_centered_padding(shape):
    image = np.full(shape, 255, dtype=np.uint8)

    result = preprocessing.resize_image(image, (112, 112))[0]

    assert result.shape == (112, 112, 3)
    if shape[0] == shape[1]:
        np.testing.assert_array_equal(result, np.ones_like(result))
    elif shape[0] < shape[1]:
        assert not result[:28].any()
        assert not result[84:].any()
        np.testing.assert_array_equal(result[28:84], np.ones_like(result[28:84]))
    else:
        assert not result[:, :28].any()
        assert not result[:, 84:].any()
        np.testing.assert_array_equal(result[:, 28:84], np.ones_like(result[:, 28:84]))


@pytest.mark.parametrize("shape", [(1, 1000, 3), (1000, 1, 3), (20, 30, 3)])
def test_represent_accepts_nonempty_thin_images(shape):
    image = np.full(shape, [20, 100, 220], dtype=np.uint8)

    results = DeepFace.represent(image, model_name="SFace", detector_backend="skip")

    assert len(results) == 1
    assert len(results[0]["embedding"]) == 128
    assert np.isfinite(results[0]["embedding"]).all()
