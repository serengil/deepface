# built-in dependencies
import os
import hashlib
import importlib
from typing import List

# 3rd party dependencies
import pytest

# project dependencies
from deepface.modules import modeling
from deepface.commons.logger import Logger

logger = Logger()

# emotion model is small enough to download its weights from every source in each run.
# its module is the one implemented for the backend engine in use.
EMOTION = importlib.import_module(
    modeling.get_model_class(task="facial_attribute", model_name="Emotion").__module__
)
SOURCES: List[str] = (
    [EMOTION.WEIGHTS_URL] if isinstance(EMOTION.WEIGHTS_URL, str) else list(EMOTION.WEIGHTS_URL)
)
FILE_NAME = SOURCES[0].split("/")[-1]

MISSING_URL = (
    f"https://github.com/serengil/deepface_models/releases/download/v1.0/missing_{FILE_NAME}"
)


@pytest.fixture
def deepface_home(tmp_path, monkeypatch):
    # download the weights to a temporary location instead of the real deepface home
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    os.makedirs(tmp_path / ".deepface" / "weights")
    yield tmp_path


def __weight_file(home) -> str:
    return os.path.join(str(home), ".deepface", "weights", FILE_NAME)


def __sha256(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.parametrize("url", SOURCES)
def test_each_source_is_downloadable(deepface_home, url):
    model = EMOTION.load_model(url=url)
    assert model is not None
    assert os.path.getsize(__weight_file(deepface_home)) > 0
    logger.info(f"✅ emotion weights downloaded and loaded from {url}")


def test_sources_serve_identical_weights(tmp_path, monkeypatch):
    if len(SOURCES) < 2:
        pytest.skip(f"emotion model has a single weight source - {SOURCES[0]}")

    hashes = set()
    for idx, url in enumerate(SOURCES):
        home = tmp_path / str(idx)
        os.makedirs(home / ".deepface" / "weights")
        monkeypatch.setenv("DEEPFACE_HOME", str(home))
        EMOTION.load_model(url=url)
        hashes.add(__sha256(__weight_file(home)))

    assert len(hashes) == 1, f"weight sources serve different files - {SOURCES}"
    logger.info("✅ emotion weight sources serve identical files")


def test_first_source_succeeds(deepface_home):
    model = EMOTION.load_model(url=SOURCES)
    assert model is not None
    assert os.path.isfile(__weight_file(deepface_home))
    logger.info("✅ emotion weights downloaded from the first source")


def test_first_source_fails_backup_succeeds(deepface_home):
    model = EMOTION.load_model(url=[MISSING_URL] + SOURCES)
    assert model is not None
    assert os.path.isfile(__weight_file(deepface_home))
    logger.info("✅ emotion weights downloaded from the backup source")


def test_all_sources_fail(deepface_home):
    with pytest.raises(ValueError, match="An exception occurred while downloading"):
        EMOTION.load_model(url=[MISSING_URL, MISSING_URL.replace("missing_", "absent_")])

    # error pages must not be left behind as if they were weight files
    assert not os.path.isfile(__weight_file(deepface_home))
    logger.info("✅ downloading emotion weights failed for all sources as expected")
