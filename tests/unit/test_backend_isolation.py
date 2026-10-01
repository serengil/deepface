# built-in dependencies
import json
import os
import subprocess
import sys

# 3rd party dependencies
import pytest

# project dependencies
from deepface.commons import backend_utils
from deepface.commons.logger import Logger

logger = Logger()

# top level modules each backend engine brings in
FRAMEWORK_MODULES = {
    backend_utils.TENSORFLOW: ["tensorflow", "keras", "tf_keras"],
    backend_utils.PYTORCH: ["torch"],
    backend_utils.ONNX: ["onnxruntime"],
}

# the backend engine is picked once and cached, and this process may have imported a
# framework already. so, verify runs in a fresh interpreter to see what it really imports.
SCRIPT = """
import json, sys
from deepface import DeepFace
from deepface.commons import backend_utils

result = DeepFace.verify(
    "dataset/img1.jpg", "dataset/img2.jpg", model_name="VGG-Face", detector_backend="opencv"
)
print(json.dumps({
    "backend": backend_utils.get_backend_engine(),
    "verified": result["verified"],
    "modules": sorted({name.split(".")[0] for name in sys.modules}),
}))
"""


@pytest.mark.parametrize("backend", backend_utils.BACKENDS)
def test_verify_imports_only_its_backend_engine(backend: str):
    if not backend_utils.is_backend_available(backend):
        pytest.skip(f"{backend} backend engine is not installed")

    env = {**os.environ, backend_utils.BACKEND_ENGINE_ENV_VAR: backend}
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    # deepface may log to stdout, the result is the last line
    output = json.loads(completed.stdout.strip().splitlines()[-1])
    assert output["backend"] == backend
    assert output["verified"] is True

    imported = set(output["modules"])

    # the engine in use must really be the one running verify
    assert FRAMEWORK_MODULES[backend][0] in imported

    for other, modules in FRAMEWORK_MODULES.items():
        if other == backend:
            continue
        leaked = imported.intersection(modules)
        assert not leaked, f"{backend} backend engine imported {other}'s {sorted(leaked)}"

    logger.info(f"✅ verify on {backend} does not import other backend engines")
