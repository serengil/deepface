# 3rd party dependencies
import pytest
import cv2

# project dependencies
from deepface import DeepFace
from deepface.modules.verification import find_distance
from deepface.commons.logger import Logger

logger = Logger()

models = ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "GhostFaceNet"]
metrics = ["cosine", "euclidean", "euclidean_l2", "angular"]
detectors = ["opencv", "mtcnn"]


def test_different_facial_recognition_models():
    dataset = [
        ["dataset/img1.jpg", "dataset/img2.jpg", True],
        ["dataset/img5.jpg", "dataset/img6.jpg", True],
        ["dataset/img6.jpg", "dataset/img7.jpg", True],
        ["dataset/img8.jpg", "dataset/img9.jpg", True],
        ["dataset/img1.jpg", "dataset/img11.jpg", True],
        ["dataset/img2.jpg", "dataset/img11.jpg", True],
        ["dataset/img1.jpg", "dataset/img3.jpg", False],
        ["dataset/img2.jpg", "dataset/img3.jpg", False],
        ["dataset/img6.jpg", "dataset/img8.jpg", False],
        ["dataset/img6.jpg", "dataset/img9.jpg", False],
    ]

    expected_coverage = 97.53  # human level accuracy on LFW
    successful_tests = 0
    unsuccessful_tests = 0
    for model in models:
        for metric in metrics:
            for instance in dataset:
                img1 = instance[0]
                img2 = instance[1]
                result = instance[2]

                resp_obj = DeepFace.verify(img1, img2, model_name=model, distance_metric=metric)

                prediction = resp_obj["verified"]
                distance = round(resp_obj["distance"], 2)
                threshold = resp_obj["threshold"]

                if prediction is result:
                    test_result_label = "✅"
                    successful_tests += 1
                else:
                    test_result_label = "❌"
                    unsuccessful_tests += 1

                if prediction is True:
                    classified_label = "same person"
                else:
                    classified_label = "different persons"

                img1_alias = img1.split("/", maxsplit=1)[-1]
                img2_alias = img2.split("/", maxsplit=1)[-1]

                logger.debug(
                    f"{test_result_label} Pair {img1_alias}-{img2_alias}"
                    f" is {classified_label} based on {model}-{metric}"
                    f" (Distance: {distance}, Threshold: {threshold})",
                )

    coverage_score = (100 * successful_tests) / (successful_tests + unsuccessful_tests)
    assert (
        coverage_score > expected_coverage
    ), f"⛔ facial recognition models test failed with {coverage_score} score"

    logger.info(f"✅ facial recognition models test passed with {coverage_score}")


def test_different_face_detectors():
    for detector in detectors:
        res = DeepFace.verify("dataset/img1.jpg", "dataset/img2.jpg", detector_backend=detector)
        assert isinstance(res, dict)
        assert "verified" in res.keys()
        assert res["verified"] in [True, False]
        assert "distance" in res.keys()
        assert "threshold" in res.keys()
        assert "model" in res.keys()
        assert "detector_backend" in res.keys()
        assert "similarity_metric" in res.keys()
        assert "facial_areas" in res.keys()
        assert "img1" in res["facial_areas"].keys()
        assert "img2" in res["facial_areas"].keys()
        assert "x" in res["facial_areas"]["img1"].keys()
        assert "y" in res["facial_areas"]["img1"].keys()
        assert "w" in res["facial_areas"]["img1"].keys()
        assert "h" in res["facial_areas"]["img1"].keys()
        assert "x" in res["facial_areas"]["img2"].keys()
        assert "y" in res["facial_areas"]["img2"].keys()
        assert "w" in res["facial_areas"]["img2"].keys()
        assert "h" in res["facial_areas"]["img2"].keys()
        assert "confidence" in res.keys()
        assert isinstance(res["confidence"], (float, int))
        assert 0 <= res["confidence"] <= 100, "Confidence should be between 0 and 100"
        if res["verified"]:
            assert (
                51 <= res["confidence"] <= 100
            ), "Confidence for same person should be between 51 and 100"
        else:
            assert (
                0 <= res["confidence"] <= 49
            ), "Confidence for different persons should be between 0 and 49"
        logger.info(f"✅ test verify for {detector} backend done")


def test_verify_for_preloaded_image():
    img1 = cv2.imread("dataset/img1.jpg")
    img2 = cv2.imread("dataset/img2.jpg")
    res = DeepFace.verify(img1, img2)
    assert res["verified"] is True
    logger.info("✅ test verify for pre-loaded image done")


def test_verify_for_precalculated_embeddings():
    model_name = "Facenet"

    img1_path = "dataset/img1.jpg"
    img2_path = "dataset/img2.jpg"

    img1_embedding = DeepFace.represent(img_path=img1_path, model_name=model_name)[0]["embedding"]
    img2_embedding = DeepFace.represent(img_path=img2_path, model_name=model_name)[0]["embedding"]

    result = DeepFace.verify(
        img1_path=img1_embedding, img2_path=img2_embedding, model_name=model_name, silent=True
    )

    assert result["verified"] is True
    assert result["distance"] < result["threshold"]
    assert result["model"] == model_name
    assert result["facial_areas"]["img1"] is not None
    assert result["facial_areas"]["img2"] is not None

    assert isinstance(result["facial_areas"]["img1"], dict)
    assert isinstance(result["facial_areas"]["img2"], dict)

    assert "x" in result["facial_areas"]["img1"].keys()
    assert "y" in result["facial_areas"]["img1"].keys()
    assert "w" in result["facial_areas"]["img1"].keys()
    assert "h" in result["facial_areas"]["img1"].keys()
    assert "left_eye" in result["facial_areas"]["img1"].keys()
    assert "right_eye" in result["facial_areas"]["img1"].keys()

    assert "x" in result["facial_areas"]["img2"].keys()
    assert "y" in result["facial_areas"]["img2"].keys()
    assert "w" in result["facial_areas"]["img2"].keys()
    assert "h" in result["facial_areas"]["img2"].keys()
    assert "left_eye" in result["facial_areas"]["img2"].keys()
    assert "right_eye" in result["facial_areas"]["img2"].keys()

    logger.info("✅ test verify for pre-calculated embeddings done")


def test_verify_with_precalculated_embeddings_for_incorrect_model():
    # generate embeddings with VGG (default)
    img1_path = "dataset/img1.jpg"
    img2_path = "dataset/img2.jpg"
    img1_embedding = DeepFace.represent(img_path=img1_path)[0]["embedding"]
    img2_embedding = DeepFace.represent(img_path=img2_path)[0]["embedding"]

    with pytest.raises(
        ValueError,
        match="embeddings of Facenet should have 128 dimensions, but 1-th image has 4096 dimensions input",
    ):
        _ = DeepFace.verify(
            img1_path=img1_embedding, img2_path=img2_embedding, model_name="Facenet", silent=True
        )

    logger.info("✅ test verify with pre-calculated embeddings for incorrect model done")


def test_verify_for_broken_embeddings():
    img1_embeddings = ["a", "b", "c"]
    img2_embeddings = [1, 2, 3]

    with pytest.raises(
        ValueError,
        match="When passing img1_path as a list, ensure that all its items are of type float.",
    ):
        _ = DeepFace.verify(img1_path=img1_embeddings, img2_path=img2_embeddings)
    logger.info("✅ test verify for broken embeddings content is done")


def test_verify_for_nested_embeddings():
    """
    batch embeddings not supported
    """
    img1_embeddings = [[1, 2, 3], [4, 5, 6]]
    img2_path = "dataset/img1.jpg"

    with pytest.raises(
        ValueError,
        match="When passing img1_path as a list, ensure that all its items are of type float",
    ):
        _ = DeepFace.verify(img1_path=img1_embeddings, img2_path=img2_path)

    logger.info("✅ test verify for nested embeddings is done")


def test_compability_of_verify_and_represent():
    """
    verify and represent should be compatible
        because of rgb bgr conversion, this is confusing
    """
    img1 = "dataset/img1.jpg"
    img2 = "dataset/img2.jpg"

    resp_obj = DeepFace.verify(
        img1, img2, model_name="Facenet", detector_backend="mtcnn", distance_metric="cosine"
    )
    alpha = resp_obj["distance"]

    img1_repr = DeepFace.represent(img1, model_name="Facenet", detector_backend="mtcnn")[0][
        "embedding"
    ]
    img2_repr = DeepFace.represent(img2, model_name="Facenet", detector_backend="mtcnn")[0][
        "embedding"
    ]

    beta = find_distance(
        alpha_embedding=img1_repr,
        beta_embedding=img2_repr,
        distance_metric="cosine",
    )

    assert abs(alpha - beta) < 1e-6, f"Distance mismatch: {alpha} (verify) vs {beta} (represent)"

    logger.info("✅ test compatibility of verify and represent is done")


def test_confidence():
    for distance_metric in metrics:
        result = DeepFace.verify(
            "dataset/img1.jpg",
            "dataset/img2.jpg",
            model_name="Facenet",
            detector_backend="mtcnn",
            distance_metric=distance_metric,
        )
        assert (
            result["confidence"] >= 51
        ), f"Confidence should be >= 51 for same person, got {result['confidence']}"

        result = DeepFace.verify(
            "dataset/img1.jpg",
            "dataset/img8.jpg",
            model_name="Facenet",
            detector_backend="mtcnn",
            distance_metric=distance_metric,
        )
        assert (
            result["confidence"] <= 49
        ), f"Confidence should be <= 49 for different persons, got {result['confidence']}"
        logger.info(f"✅ test confidence for {distance_metric} metric is done")
