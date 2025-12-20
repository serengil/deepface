# built-in dependencies
import os

# 3rd party dependencies
import cv2

# project dependencies
from deepface import DeepFace
from deepface.modules import verification
from deepface.commons.logger import Logger

logger = Logger()


threshold = verification.find_threshold(model_name="VGG-Face", distance_metric="cosine")


def test_find_with_exact_path():
    img_path = os.path.join("dataset", "img1.jpg")
    results = DeepFace.find(img_path=img_path, db_path="dataset", silent=True, batched=True)
    assert len(results) > 0
    required_keys = set(
        [
            "identity",
            "distance",
            "threshold",
            "hash",
            "target_x",
            "target_y",
            "target_w",
            "target_h",
            "source_x",
            "source_y",
            "source_w",
            "source_h",
        ]
    )
    for result in results:
        assert isinstance(result, list)

        found_image_itself = False
        for face in result:
            assert isinstance(face, dict)
            assert set(face.keys()) == required_keys
            if face["identity"] == img_path:
                # validate reproducability
                assert face["distance"] < threshold
                # one is img1.jpg itself
                found_image_itself = True
        assert found_image_itself

    assert len(results[0]) > 1

    logger.info("✅ test find for exact path done")


def test_batched_find_with_similarity_search():
    angelinas = [
        os.path.join("dataset", "img1.jpg"),
        os.path.join("dataset", "img2.jpg"),
        os.path.join("dataset", "img4.jpg"),
        os.path.join("dataset", "img5.jpg"),
        os.path.join("dataset", "img6.jpg"),
        os.path.join("dataset", "img7.jpg"),
        os.path.join("dataset", "img10.jpg"),
        os.path.join("dataset", "img11.jpg"),
        os.path.join("dataset", "img11_reflection.jpg"),
        os.path.join("dataset", "couple.jpg"),
        os.path.join("dataset", "selfie-many-people.jpg"),
    ]

    img_path = os.path.join("dataset", "img1.jpg")
    k = len(angelinas) + 5
    verification_threshold = verification.find_threshold(
        model_name="VGG-Face", distance_metric="cosine"
    )
    results = DeepFace.find(
        img_path=img_path, db_path="dataset", silent=True, batched=True, similarity_search=True, k=k
    )
    assert isinstance(results, list)
    assert len(results) > 0
    for inner_results in results:
        assert isinstance(inner_results, list)

        assert len(inner_results) <= k
        verified = 0
        similar = 0
        for result in inner_results:
            assert isinstance(result, dict)
            if result["identity"] in angelinas:
                verified += 1
                assert result["distance"] <= verification_threshold
            else:
                similar += 1
                assert result["distance"] > verification_threshold

        assert verified > 0
        assert similar > 0

    logger.info("✅ test batched find with similarity search done")


def test_find_with_array_input():
    img_path = os.path.join("dataset", "img1.jpg")
    img1 = cv2.imread(img_path)
    results = DeepFace.find(img1, db_path="dataset", silent=True, batched=True)
    assert len(results) > 0
    for result in results:
        assert isinstance(result, list)

        found_image_itself = False
        for face in result:
            assert isinstance(face, dict)
            if face["identity"] == img_path:
                # validate reproducability
                assert face["distance"] < threshold
                # one is img1.jpg itself
                found_image_itself = True
        assert found_image_itself

    assert len(results[0]) > 1

    logger.info("✅ test find for array input done")


def test_find_with_extracted_faces():
    img_path = os.path.join("dataset", "img1.jpg")
    face_objs = DeepFace.extract_faces(img_path)
    img = face_objs[0]["face"]
    results = DeepFace.find(
        img, db_path="dataset", detector_backend="skip", silent=True, batched=True
    )
    assert len(results) > 0
    for result in results:
        assert isinstance(result, list)

        found_image_itself = False
        for face in result:
            assert isinstance(face, dict)
            if face["identity"] == img_path:
                # validate reproducability
                assert face["distance"] < threshold
                # one is img1.jpg itself
                found_image_itself = True
        assert found_image_itself

    assert len(results[0]) > 1
    logger.info("✅ test find for extracted face input done")


def test_filetype_for_find():
    """
    only images as jpg and png can be loaded into database
    """
    img_path = os.path.join("dataset", "img1.jpg")
    results = DeepFace.find(img_path=img_path, db_path="dataset", silent=True, batched=True)

    result = results[0]

    assert not any(face["identity"] == "dataset/img47.jpg" for face in result)

    logger.info("✅ test wrong filetype done")
