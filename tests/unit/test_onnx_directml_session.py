# built-in dependencies
import sys
from types import SimpleNamespace
from unittest.mock import Mock

# 3rd party dependencies
import pytest

# project dependencies
from deepface.commons import onnx_utils


@pytest.mark.parametrize(
    "requested, available, expected_providers, use_directml",
    [
        ("directml", ["DmlExecutionProvider", "CPUExecutionProvider"],
         ["DmlExecutionProvider", "CPUExecutionProvider"], True),
        ("cpu,directml", ["DmlExecutionProvider", "CPUExecutionProvider"],
         ["CPUExecutionProvider", "DmlExecutionProvider"], True),
        ("directml", ["CPUExecutionProvider"], ["CPUExecutionProvider"], False),
        ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"],
         ["CUDAExecutionProvider", "CPUExecutionProvider"], False),
    ],
)
def test_session_options_match_resolved_providers(monkeypatch, requested, available,
                                                expected_providers, use_directml):
    # These options are required by DirectML even when it is not first in the list.
    options = SimpleNamespace(enable_mem_pattern=True, execution_mode="parallel")
    runtime = SimpleNamespace(
        get_available_providers=lambda: available,
        SessionOptions=lambda: options,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=Mock(),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setenv(onnx_utils.PROVIDERS_ENV_VAR, requested)
    monkeypatch.setattr(onnx_utils.weight_utils, "download_weights_if_necessary",
                        lambda **kwargs: "model.onnx")

    session = onnx_utils.build_session("model.onnx", "https://example.com/model.onnx")

    runtime.InferenceSession.assert_called_once_with(
        "model.onnx", sess_options=options, providers=expected_providers
    )
    assert session is runtime.InferenceSession.return_value
    assert options.log_severity_level == 3
    assert options.enable_mem_pattern is not use_directml
    assert options.execution_mode == ("sequential" if use_directml else "parallel")
