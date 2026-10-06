# 3rd party dependencies
import pytest

# project dependencies
from deepface.models.face_detection import Ssd
from deepface.commons.logger import Logger

logger = Logger()


def test_ssd_requires_opencv_below_5(monkeypatch):
    monkeypatch.setattr(Ssd.cv2, "__version__", "5.0.0")

    def download_weights_if_necessary(**kwargs):
        raise AssertionError("weights must not be downloaded for unsupported opencv")

    monkeypatch.setattr(
        Ssd.weight_utils, "download_weights_if_necessary", download_weights_if_necessary
    )

    with pytest.raises(ValueError, match="Ssd requires opencv-python < 5 but you have 5.0.0"):
        Ssd.SsdClient()
    logger.info("✅ test ssd requires opencv below 5 done")
