# built-in dependencies
import io
import cv2
import pytest
import numpy as np
import pytest

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
    default_embedding_objs = DeepFace.represent(img_path)
    io_embedding_objs = DeepFace.represent(open(img_path, 'rb'))
    assert default_embedding_objs == io_embedding_objs

    # Confirm non-seekable io objects are handled properly
    io_obj = io.BytesIO(open(img_path, 'rb').read())
    io_obj.seek = None
    no_seek_io_embedding_objs = DeepFace.represent(io_obj)
    assert default_embedding_objs == no_seek_io_embedding_objs

    # Confirm non-image io objects raise exceptions
    with pytest.raises(ValueError, match='Failed to decode image'):
        DeepFace.represent(io.BytesIO(open(r'../requirements.txt', 'rb').read()))

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


@pytest.mark.parametrize("model_name", [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "Dlib", 
    "ArcFace", 
    "SFace", 
    "GhostFaceNet"
])
def test_batched_represent(model_name):
    img_paths = [
        "dataset/img1.jpg",
        "dataset/img2.jpg",
        "dataset/img3.jpg",
        "dataset/img4.jpg",
        "dataset/img5.jpg",
    ]

    embedding_objs = DeepFace.represent(img_path=img_paths, model_name=model_name)
    assert len(embedding_objs) == len(img_paths), f"Expected {len(img_paths)} embeddings, got {len(embedding_objs)}"

    if model_name == "VGG-Face":
        for embedding_obj in embedding_objs:
            embedding = embedding_obj["embedding"]
            logger.debug(f"Function returned {len(embedding)} dimensional vector")
            assert len(embedding) == 4096, f"Expected embedding of length 4096, got {len(embedding)}"

    embedding_objs_one_by_one = [
        embedding_obj
        for img_path in img_paths
        for embedding_obj in DeepFace.represent(img_path=img_path, model_name=model_name) 
    ]
    for embedding_obj_one_by_one, embedding_obj in zip(embedding_objs_one_by_one, embedding_objs):
        assert np.allclose(
            embedding_obj_one_by_one["embedding"], 
            embedding_obj["embedding"],
            rtol=1e-2,
            atol=1e-2
        ), "Embeddings do not match within tolerance"

    logger.info(f"✅ test batch represent function for model {model_name} done")
