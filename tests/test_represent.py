# built-in dependencies
import io
import cv2

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


def test_standard_represent():
    img_path = "dataset/img1.jpg"
    embedding_objs = DeepFace.represent(img_path)
    for embedding_obj in embedding_objs:
        embedding = embedding_obj["embedding"]
        logger.debug(f"Function returned {len(embedding)} dimensional vector")
        assert len(embedding) == 4096
    logger.info("✅ test standard represent function done")


def test_standard_represent_with_io_object():
    img_path = "dataset/img1.jpg"
    defualt_embedding_objs = DeepFace.represent(img_path)
    io_embedding_objs = DeepFace.represent(open(img_path, 'rb'))
    assert defualt_embedding_objs == io_embedding_objs

    # Confirm non-seekable io objects are handled properly
    io_obj = io.BytesIO(open(img_path, 'rb').read())
    io_obj.seek = None
    no_seek_io_embedding_objs = DeepFace.represent(io_obj)
    assert defualt_embedding_objs == no_seek_io_embedding_objs

    logger.info("✅ test standard represent with io object function done")


def test_represent_for_skipped_detector_backend_with_image_path():
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
    logger.info("✅ test represent function for skipped detector and image path input backend done")


def test_represent_for_skipped_detector_backend_with_preloaded_image():
    face_img = "dataset/img5.jpg"
    img = cv2.imread(face_img)
    img_objs = DeepFace.represent(img_path=img, detector_backend="skip")
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
    logger.info("✅ test represent function for skipped detector and preloaded image done")


def test_max_faces():
    # confirm that input image has more than one face
    results = DeepFace.represent(img_path="dataset/couple.jpg")
    assert len(results) > 1

    # test it with max faces arg
    max_faces = 1
    results = DeepFace.represent(img_path="dataset/couple.jpg", max_faces=max_faces)
    assert len(results) == max_faces
