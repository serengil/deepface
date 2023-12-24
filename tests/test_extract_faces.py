from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger("tests/test_extract_faces.py")


def test_different_detectors():
    detectors = ["opencv", "mtcnn"]

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
