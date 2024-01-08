import pytest
import numpy as np
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger("tests/test_enforce_detection.py")


def test_enabled_enforce_detection_for_non_facial_input():
    black_img = np.zeros([224, 224, 3])

    with pytest.raises(ValueError, match="Face could not be detected."):
        DeepFace.represent(img_path=black_img)

    with pytest.raises(ValueError, match="Face could not be detected."):
        DeepFace.verify(img1_path=black_img, img2_path=black_img)

    logger.info("✅ enabled enforce detection with non facial input tests done")


def test_disabled_enforce_detection_for_non_facial_input_on_represent():
    black_img = np.zeros([224, 224, 3])
    objs = DeepFace.represent(img_path=black_img, enforce_detection=False)

    assert isinstance(objs, list)
    assert len(objs) > 0
    assert isinstance(objs[0], dict)
    assert "embedding" in objs[0].keys()
    assert "facial_area" in objs[0].keys()
    assert isinstance(objs[0]["facial_area"], dict)
    assert "x" in objs[0]["facial_area"].keys()
    assert "y" in objs[0]["facial_area"].keys()
    assert "w" in objs[0]["facial_area"].keys()
    assert "h" in objs[0]["facial_area"].keys()
    assert isinstance(objs[0]["embedding"], list)
    assert len(objs[0]["embedding"]) == 4096  # embedding of VGG-Face

    logger.info("✅ disabled enforce detection with non facial input test for represent tests done")


def test_disabled_enforce_detection_for_non_facial_input_on_verify():
    black_img = np.zeros([224, 224, 3])
    obj = DeepFace.verify(img1_path=black_img, img2_path=black_img, enforce_detection=False)
    assert isinstance(obj, dict)

    logger.info("✅ disabled enforce detection with non facial input test for verify tests done")
