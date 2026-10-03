"""NumPy batches use the same analysis path at every batch size."""

import numpy as np
import pytest

from deepface.modules import demography, modeling


class FixedAgeModel:
    def predict(self, img):
        assert img.shape == (1, 224, 224, 3)
        return 32


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_analyze_numpy_batch_matches_list_input(monkeypatch, batch_size):
    images = np.full((batch_size, 8, 12, 3), [20, 80, 220], dtype=np.uint8)
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: FixedAgeModel())
    options = dict(actions=("age",), detector_backend="skip", silent=True)

    expected = demography.analyze(list(images), **options)
    result = demography.analyze(images, **options)

    assert result == expected
    assert len(result) == batch_size
    for faces in result:
        assert faces[0]["age"] == 32
        assert faces[0]["region"]["x"] == 0


def test_analyze_single_image_keeps_flat_result(monkeypatch):
    image = np.full((8, 12, 3), [20, 80, 220], dtype=np.uint8)
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: FixedAgeModel())
    result = demography.analyze(
        image, actions=("age",), detector_backend="skip", silent=True
    )
    assert len(result) == 1
    assert result[0]["age"] == 32
