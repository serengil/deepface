import numpy as np
import pytest
from deepface.modules.detection import extract_faces, DetectedFace, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()

def is_valid_landmark(coord, width, height):
    if coord is None:
        return False
    if not (isinstance(coord, (tuple, list)) and len(coord) == 2):
        return False
    x, y = coord
    return 0 <= x < width and 0 <= y < height

def sanitize_landmarks(region, width, height):
    landmarks = {
        "left_eye": region.left_eye,
        "right_eye": region.right_eye,
        "nose": region.nose,
        "mouth_left": region.mouth_left,
        "mouth_right": region.mouth_right,
    }
    for key, value in landmarks.items():
        if not is_valid_landmark(value, width, height):
            landmarks[key] = None
    return landmarks

def test_sanitize_landmarks():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    height, width = img.shape[:2]
    region = FacialAreaRegion(
        x=10, y=10, w=50, h=50,
        left_eye=(-5, 20),  # invalid
        right_eye=(20, 200),  # invalid
        nose=(30, 30),  # valid
        mouth_left=(150, 20),  # invalid
        mouth_right=(20, -10),  # invalid
        confidence=0.9
    )
    landmarks = sanitize_landmarks(region, width, height)
    logger.info(f"Sanitized landmarks: {landmarks}")
    assert landmarks["left_eye"] is None
    assert landmarks["right_eye"] is None
    assert landmarks["nose"] == (30, 30)
    assert landmarks["mouth_left"] is None
    assert landmarks["mouth_right"] is None
    logger.info("Test passed: Invalid landmarks are sanitized to None.")

def test_extract_faces_sanitizes_landmarks(monkeypatch):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    facial_area = FacialAreaRegion(
        x=10, y=10, w=50, h=50,
        left_eye=(-5, 20),  # invalid
        right_eye=(20, 200),  # invalid
        nose=(30, 30),  # valid
        mouth_left=(150, 20),  # invalid
        mouth_right=(20, -10),  # invalid
        confidence=0.9
    )
    detected_face = DetectedFace(img=img, facial_area=facial_area, confidence=0.9)
    monkeypatch.setattr("deepface.modules.detection.detect_faces", lambda *args, **kwargs: [detected_face])
    result = extract_faces(img, detector_backend="opencv", enforce_detection=False)
    facial_area_out = result[0]["facial_area"]
    logger.info(f"Output facial_area: {facial_area_out}")
    assert facial_area_out["left_eye"] is None
    assert facial_area_out["right_eye"] is None
    assert facial_area_out.get("nose") == (30, 30)
    assert facial_area_out.get("mouth_left") is None
    assert facial_area_out.get("mouth_right") is None 