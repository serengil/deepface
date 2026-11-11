# built-in dependencies
from collections import UserList
from types import SimpleNamespace
from unittest.mock import Mock

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules import representation


@pytest.mark.parametrize("container", [tuple, UserList])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_documented_sequences_behave_like_list_batches(monkeypatch, container, count):
    def forward(images):
        embeddings = images.mean(axis=(1, 2)).tolist()
        return embeddings[0] if len(embeddings) == 1 else embeddings

    model = SimpleNamespace(input_shape=(4, 4), forward=Mock(side_effect=forward))
    build_model = Mock(return_value=model)
    monkeypatch.setattr(representation.modeling, "build_model", build_model)
    images = [np.full((4, 4, 3), 30 + index * 20, dtype=np.uint8) for index in range(count)]
    expected = representation.represent(images, detector_backend="skip")
    actual = representation.represent(container(images), detector_backend="skip")

    assert actual == expected
    assert len(actual) == count
    if count == 0:
        build_model.assert_not_called()
    else:
        assert all(isinstance(faces, list) and len(faces) == 1 for faces in actual)
