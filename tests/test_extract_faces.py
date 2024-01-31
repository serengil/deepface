import numpy as np
import pytest
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger("tests/test_extract_faces.py")

detectors = ["opencv", "mtcnn"]


def test_different_detectors():
    for detector in detectors:
        img_objs = DeepFace.extract_faces(img_path="dataset/img11.jpg", detector_backend=detector)
        for img_obj in img_objs:
            assert "face" in img_obj.keys()
            assert "facial_area" in img_obj.keys()
            assert isinstance(img_obj["facial_area"], dict)
            assert "x" in img_obj["facial_area"].keys()
            assert "y" in img_obj["facial_area"].keys()
            assert "w" in img_obj["facial_area"].keys()
            assert "h" in img_obj["facial_area"].keys()
            assert "confidence" in img_obj.keys()

            img = img_obj["face"]
            assert img.shape[0] > 0 and img.shape[1] > 0
        logger.info(f"✅ extract_faces for {detector} backend test is done")


def test_backends_for_enforced_detection_with_non_facial_inputs():
    black_img = np.zeros([224, 224, 3])
    for detector in detectors:
        with pytest.raises(ValueError):
            _ = DeepFace.extract_faces(img_path=black_img, detector_backend=detector)
    logger.info("✅ extract_faces for enforced detection and non-facial image test is done")


def test_backends_for_not_enforced_detection_with_non_facial_inputs():
    black_img = np.zeros([224, 224, 3])
    for detector in detectors:
        objs = DeepFace.extract_faces(
            img_path=black_img, detector_backend=detector, enforce_detection=False
        )
        assert objs[0]["face"].shape == (224, 224, 3)
    logger.info("✅ extract_faces for not enforced detection and non-facial image test is done")
