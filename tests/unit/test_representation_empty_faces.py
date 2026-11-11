# built-in dependencies
from types import SimpleNamespace
from unittest.mock import Mock

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules import representation


@pytest.mark.parametrize("batch", [False, True])
def test_all_empty_detections_return_empty_results(monkeypatch, batch):
    model = SimpleNamespace(input_shape=(4, 4), forward=Mock())
    monkeypatch.setattr(representation.modeling, "build_model", lambda **kwargs: model)
    monkeypatch.setattr(representation.detection, "extract_faces", lambda **kwargs: [])
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    result = representation.represent([image, image] if batch else image,
                                      enforce_detection=False)

    assert result == ([[], []] if batch else [])
    model.forward.assert_not_called()


def test_mixed_batch_preserves_empty_image_position(monkeypatch):
    model = SimpleNamespace(input_shape=(4, 4), forward=Mock(return_value=[1.0, 2.0]))
    monkeypatch.setattr(representation.modeling, "build_model", lambda **kwargs: model)
    image = np.ones((4, 4, 3), dtype=np.uint8)
    extracted = Mock(side_effect=[[], [{"face": image, "confidence": 1.0,
                                       "facial_area": {"x": 0, "y": 0, "w": 4, "h": 4}}]])
    monkeypatch.setattr(representation.detection, "extract_faces", extracted)

    result = representation.represent([image, image], enforce_detection=False)

    assert result[0] == []
    assert result[1][0]["embedding"] == [1.0, 2.0]
    model.forward.assert_called_once()
