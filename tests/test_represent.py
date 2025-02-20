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
    # type should be list of dict
    assert isinstance(embedding_objs, list)

    for embedding_obj in embedding_objs:
        assert isinstance(embedding_obj, dict)

        embedding = embedding_obj["embedding"]
        logger.debug(f"Function returned {len(embedding)} dimensional vector")
        assert len(embedding) == 4096
    logger.info("✅ test standard represent function done")


def test_standard_represent_with_io_object():
    img_path = "dataset/img1.jpg"
    default_embedding_objs = DeepFace.represent(img_path)
    io_embedding_objs = DeepFace.represent(open(img_path, "rb"))
    assert default_embedding_objs == io_embedding_objs

    # Confirm non-seekable io objects are handled properly
    io_obj = io.BytesIO(open(img_path, "rb").read())
    io_obj.seek = None
    no_seek_io_embedding_objs = DeepFace.represent(io_obj)
    assert default_embedding_objs == no_seek_io_embedding_objs

    # Confirm non-image io objects raise exceptions
    with pytest.raises(ValueError, match="Failed to decode image"):
        DeepFace.represent(io.BytesIO(open(r"../requirements.txt", "rb").read()))

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


def test_represent_for_preloaded_image():
    face_img = "dataset/img5.jpg"
    img = cv2.imread(face_img)
    img_objs = DeepFace.represent(img_path=img)
    # type should be list of dict
    assert isinstance(img_objs, list)
    assert len(img_objs) >= 1

    for img_obj in img_objs:
        assert isinstance(img_obj, dict)
        assert "embedding" in img_obj.keys()
        assert "facial_area" in img_obj.keys()
        assert isinstance(img_obj["facial_area"], dict)
        assert "x" in img_obj["facial_area"].keys()
        assert "y" in img_obj["facial_area"].keys()
        assert "w" in img_obj["facial_area"].keys()
        assert "h" in img_obj["facial_area"].keys()
        assert "face_confidence" in img_obj.keys()
    logger.info("✅ test represent function for skipped detector and preloaded image done")


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


@pytest.mark.parametrize(
    "model_name",
    [
        "VGG-Face",
        "Facenet",
        "SFace",
    ],
)
def test_batched_represent_for_list_input(model_name):
    img_paths = [
        "dataset/img1.jpg",
        "dataset/img2.jpg",
        "dataset/img3.jpg",
        "dataset/img4.jpg",
        "dataset/img5.jpg",
        "dataset/couple.jpg",
    ]

    expected_faces = [1, 1, 1, 1, 1, 2]

    batched_embedding_objs = DeepFace.represent(img_path=img_paths, model_name=model_name)

    # type should be list of list of dict for batch input
    assert isinstance(batched_embedding_objs, list)

    assert len(batched_embedding_objs) == len(
        img_paths
    ), f"Expected {len(img_paths)} embeddings, got {len(batched_embedding_objs)}"

    # the last one has two faces
    for idx, embedding_objs in enumerate(batched_embedding_objs):
        # type should be list of list of dict for batch input
        # batched_embedding_objs was list already, embedding_objs should be list of dict
        assert isinstance(embedding_objs, list)
        for embedding_obj in embedding_objs:
            assert isinstance(embedding_obj, dict)

        assert expected_faces[idx] == len(
            embedding_objs
        ), f"{img_paths[idx]} has {expected_faces[idx]} faces, but got {len(embedding_objs)} embeddings!"

    for idx, img_path in enumerate(img_paths):
        single_embedding_objs = DeepFace.represent(img_path=img_path, model_name=model_name)
        # type should be list of dict for single input
        assert isinstance(single_embedding_objs, list)
        for embedding_obj in single_embedding_objs:
            assert isinstance(embedding_obj, dict)

        assert len(single_embedding_objs) == len(batched_embedding_objs[idx])

        for alpha, beta in zip(single_embedding_objs, batched_embedding_objs[idx]):
            assert isinstance(alpha, dict)
            assert isinstance(beta, dict)
            assert np.allclose(
                alpha["embedding"], beta["embedding"], rtol=1e-2, atol=1e-2
            ), "Embeddings do not match within tolerance"

    logger.info(f"✅ test batch represent function with string input for model {model_name} done")


@pytest.mark.parametrize(
    "model_name",
    [
        "VGG-Face",
        "Facenet",
        "SFace",
    ],
)
def test_batched_represent_for_numpy_input(model_name):
    img_paths = [
        "dataset/img1.jpg",
        "dataset/img2.jpg",
        "dataset/img3.jpg",
        "dataset/img4.jpg",
        "dataset/img5.jpg",
        "dataset/couple.jpg",
    ]
    expected_faces = [1, 1, 1, 1, 1, 2]

    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1000, 1000))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(img.shape)
        imgs.append(img)

    imgs = np.array(imgs)
    assert imgs.ndim == 4 and imgs.shape[0] == len(img_paths)

    batched_embedding_objs = DeepFace.represent(img_path=imgs, model_name=model_name)

    # type should be list of list of dict for batch input
    assert isinstance(batched_embedding_objs, list)
    for idx, batched_embedding_obj in enumerate(batched_embedding_objs):
        assert isinstance(batched_embedding_obj, list)
        # it also has to have the expected number of faces
        assert len(batched_embedding_obj) == expected_faces[idx]
        for embedding_obj in batched_embedding_obj:
            assert isinstance(embedding_obj, dict)

    # we should have the same number of embeddings as the number of images
    assert len(batched_embedding_objs) == len(img_paths)

    logger.info(f"✅ test batch represent function with numpy input for model {model_name} done")
