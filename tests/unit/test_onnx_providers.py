# 3rd party dependencies
import pytest

# project dependencies
from deepface.commons import onnx_utils
from deepface.commons.logger import Logger

logger = Logger()

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
COREML = "CoreMLExecutionProvider"


def test_default_runs_on_cpu(monkeypatch):
    monkeypatch.delenv(onnx_utils.PROVIDERS_ENV_VAR, raising=False)
    # coreml comes with the standard macos wheel, it must not be picked unless it is set
    assert onnx_utils.resolve_providers([COREML, CPU]) == [CPU]
    logger.info("✅ onnx runs on cpu by default test done")


def test_default_prefers_cuda_when_installed(monkeypatch):
    monkeypatch.delenv(onnx_utils.PROVIDERS_ENV_VAR, raising=False)
    assert onnx_utils.resolve_providers([CPU, CUDA]) == [CUDA, CPU]
    logger.info("✅ onnx prefers cuda when onnxruntime-gpu is installed test done")


def test_aliases_and_cpu_fallback(monkeypatch):
    monkeypatch.setenv(onnx_utils.PROVIDERS_ENV_VAR, " CoreML ")
    assert onnx_utils.resolve_providers([COREML, CPU]) == [COREML, CPU]
    logger.info("✅ onnx provider aliases with cpu fallback test done")


def test_order_is_kept(monkeypatch):
    monkeypatch.setenv(onnx_utils.PROVIDERS_ENV_VAR, "cpu,cuda,cuda")
    assert onnx_utils.resolve_providers([CUDA, CPU]) == [CPU, CUDA]
    logger.info("✅ onnx providers keep the given order test done")


def test_unavailable_provider_is_skipped(monkeypatch):
    monkeypatch.setenv(onnx_utils.PROVIDERS_ENV_VAR, "tensorrt,cuda")
    assert onnx_utils.resolve_providers([CPU]) == [CPU]
    logger.info("✅ unavailable onnx providers are skipped test done")


def test_warns_when_no_requested_accelerator_is_installed(monkeypatch):
    warnings = []
    monkeypatch.setattr(onnx_utils.logger, "warn", warnings.append)

    # default providers must not warn on a cpu only onnxruntime
    monkeypatch.delenv(onnx_utils.PROVIDERS_ENV_VAR, raising=False)
    onnx_utils.resolve_providers([CPU])
    assert warnings == []

    # explicitly asking for cpu only must not warn
    monkeypatch.setenv(onnx_utils.PROVIDERS_ENV_VAR, "cpu")
    onnx_utils.resolve_providers([CPU])
    assert warnings == []

    # one of the requested accelerators is installed
    monkeypatch.setenv(onnx_utils.PROVIDERS_ENV_VAR, "tensorrt,cuda")
    onnx_utils.resolve_providers([CUDA, CPU])
    assert warnings == []

    # none of the requested accelerators is installed
    onnx_utils.resolve_providers([CPU])
    assert len(warnings) == 1
    assert onnx_utils.PROVIDERS_ENV_VAR in warnings[0]
    logger.info("✅ onnx warns when no requested accelerator is installed test done")


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv(onnx_utils.PROVIDERS_ENV_VAR, "gpu")
    with pytest.raises(ValueError, match=onnx_utils.PROVIDERS_ENV_VAR):
        onnx_utils.resolve_providers([CPU])
    logger.info("✅ unknown onnx provider raises test done")
