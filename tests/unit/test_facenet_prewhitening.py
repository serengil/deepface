"""Facenet prewhitening uses the original model's adjusted standard deviation."""

from types import SimpleNamespace

import numpy as np
import pytest

from deepface import DeepFace
from deepface.modules import modeling, preprocessing


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_facenet_constant_input(dtype, value):
    image = np.full((1, 160, 160, 3), value, dtype=dtype)
    with np.errstate(invalid="raise", divide="raise"):
        result = preprocessing.normalize_input(image, normalization="Facenet")
    np.testing.assert_array_equal(result, np.zeros_like(image))
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_facenet_low_variance_input(dtype):
    image = np.zeros((1, 160, 160, 3), dtype=dtype)
    delta = 1 / 1024
    image.flat[0] = delta / 255
    result = preprocessing.normalize_input(image, normalization="Facenet")
    expected = np.full_like(image, -delta / image.size**0.5)
    expected.flat[0] = delta * (image.size - 1) / image.size**0.5
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-10)
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_facenet_high_variance_input_is_unchanged(dtype):
    image = np.linspace(0, 1, 160 * 160 * 3, dtype=dtype).reshape(1, 160, 160, 3)
    pixels = image * 255
    expected = (pixels - pixels.mean()) / pixels.std()
    result = preprocessing.normalize_input(image, normalization="Facenet")
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == dtype


def test_represent_passes_finite_prewhitened_input_to_model(monkeypatch):
    inputs = []

    def forward(image):
        inputs.append(image.copy())
        return image.mean(axis=(0, 1, 2)).tolist()

    model = SimpleNamespace(input_shape=(160, 160), forward=forward)
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: model)
    result = DeepFace.represent(
        np.zeros((160, 160, 3), dtype=np.uint8),
        model_name="Facenet",
        normalization="Facenet",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    assert inputs[0].dtype == np.float32
    assert np.isfinite(inputs[0]).all()
    np.testing.assert_array_equal(result[0]["embedding"], [0.0, 0.0, 0.0])
