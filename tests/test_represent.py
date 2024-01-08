from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger("tests/test_represent.py")


def test_standard_represent():
    img_path = "dataset/img1.jpg"
    embedding_objs = DeepFace.represent(img_path)
    for embedding_obj in embedding_objs:
        embedding = embedding_obj["embedding"]
        logger.debug(f"Function returned {len(embedding)} dimensional vector")
        assert len(embedding) == 4096
    logger.info("✅ test standard represent function done")


def test_represent_for_skipped_detector_backend():
    face_img = "dataset/img5.jpg"
    img_objs = DeepFace.represent(img_path=face_img, detector_backend="skip")
    assert len(img_objs) >= 1
    img_obj = img_objs[0]
    assert "embedding" in img_obj.keys()
    assert "facial_area" in img_obj.keys()
    assert isinstance(img_obj["facial_area"], dict)
    assert "x" in img_obj["facial_area"].keys()
    assert "y" in img_obj["facial_area"].keys()
    assert "w" in img_obj["facial_area"].keys()
    assert "h" in img_obj["facial_area"].keys()
    assert "face_confidence" in img_obj.keys()
    logger.info("✅ test represent function for skipped detector backend done")
