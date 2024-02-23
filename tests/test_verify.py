import cv2
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger("tests/test_facial_recognition_models.py")


models = ["VGG-Face", "Facenet", "Facenet512", "ArcFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]
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
        logger.info(f"✅ test verify for {detector} backend done")


def test_verify_for_preloaded_image():
    img1 = cv2.imread("dataset/img1.jpg")
    img2 = cv2.imread("dataset/img2.jpg")
    res = DeepFace.verify(img1, img2)
    assert res["verified"] is True
    logger.info("✅ test verify for pre-loaded image done")
