"""Building an underperforming facial recognition model warns once."""

import importlib

import pytest

from deepface.commons.logger import Logger


@pytest.mark.parametrize("backend", ["tf", "pytorch", "onnx"])
@pytest.mark.parametrize(
    "module_name, class_name, model_name",
    [
        ("OpenFace", "OpenFaceClient", "OpenFace"),
        ("FbDeepFace", "DeepFaceClient", "DeepFace"),
        ("DeepID", "DeepIdClient", "DeepID"),
    ],
)
def test_underperforming_warning(monkeypatch, backend, module_name, class_name, model_name):
    module = importlib.import_module(
        f"deepface.models.facial_recognition.{backend}.{module_name}"
    )
    messages = []

    def warn(self, message, show_once=False):
        assert show_once
        messages.append(message)

    monkeypatch.setattr(Logger, "warn", warn)
    monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: _Model())

    getattr(module, class_name)()

    assert len(messages) == 1
    assert model_name in messages[0] and "underperforming" in messages[0]


class _Model:
    def to(self, _device):
        return self
