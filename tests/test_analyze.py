# 3rd party dependencies
import cv2
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.models.demography import Age, Emotion, Gender, Race
from deepface.commons.logger import Logger

logger = Logger()


detectors = ["opencv", "mtcnn"]


def test_standard_analyze():
    img = "dataset/img4.jpg"
    demography_objs = DeepFace.analyze(img, silent=True)
    for demography in demography_objs:
        logger.debug(demography)
        assert type(demography) == dict
        assert demography["age"] > 20 and demography["age"] < 40
        assert demography["dominant_gender"] == "Woman"
    logger.info("✅ test standard analyze done")


def test_analyze_with_all_actions_as_tuple():
    img = "dataset/img4.jpg"
    demography_objs = DeepFace.analyze(
        img, actions=("age", "gender", "race", "emotion"), silent=True
    )

    for demography in demography_objs:
        logger.debug(f"Demography: {demography}")
        assert type(demography) == dict
        age = demography["age"]
        gender = demography["dominant_gender"]
        race = demography["dominant_race"]
        emotion = demography["dominant_emotion"]
        logger.debug(f"Age: {age}")
        logger.debug(f"Gender: {gender}")
        logger.debug(f"Race: {race}")
        logger.debug(f"Emotion: {emotion}")
        assert demography.get("age") is not None
        assert demography.get("dominant_gender") is not None
        assert demography.get("dominant_race") is not None
        assert demography.get("dominant_emotion") is not None

    logger.info("✅ test analyze for all actions as tuple done")


def test_analyze_with_all_actions_as_list():
    img = "dataset/img4.jpg"
    demography_objs = DeepFace.analyze(
        img, actions=["age", "gender", "race", "emotion"], silent=True
    )

    for demography in demography_objs:
        logger.debug(f"Demography: {demography}")
        assert type(demography) == dict
        age = demography["age"]
        gender = demography["dominant_gender"]
        race = demography["dominant_race"]
        emotion = demography["dominant_emotion"]
        logger.debug(f"Age: {age}")
        logger.debug(f"Gender: {gender}")
        logger.debug(f"Race: {race}")
        logger.debug(f"Emotion: {emotion}")
        assert demography.get("age") is not None
        assert demography.get("dominant_gender") is not None
        assert demography.get("dominant_race") is not None
        assert demography.get("dominant_emotion") is not None

    logger.info("✅ test analyze for all actions as array done")


def test_analyze_for_some_actions():
    img = "dataset/img4.jpg"
    demography_objs = DeepFace.analyze(img, ["age", "gender"], silent=True)

    for demography in demography_objs:
        assert type(demography) == dict
        age = demography["age"]
        gender = demography["dominant_gender"]

        logger.debug(f"Age: { age }")
        logger.debug(f"Gender: {gender}")

        assert demography.get("age") is not None
        assert demography.get("dominant_gender") is not None

        # these are not in actions
        assert demography.get("dominant_race") is None
        assert demography.get("dominant_emotion") is None

    logger.info("✅ test analyze for some actions done")


def test_analyze_for_preloaded_image():
    img = cv2.imread("dataset/img1.jpg")
    resp_objs = DeepFace.analyze(img, silent=True)
    for resp_obj in resp_objs:
        logger.debug(resp_obj)
        assert type(resp_obj) == dict
        assert resp_obj["age"] > 20 and resp_obj["age"] < 40
        assert resp_obj["dominant_gender"] == "Woman"

    logger.info("✅ test analyze for pre-loaded image done")


def test_analyze_for_different_detectors():
    img_paths = [
        "dataset/img1.jpg",
        "dataset/img5.jpg",
        "dataset/img6.jpg",
        "dataset/img8.jpg",
        "dataset/img1.jpg",
        "dataset/img2.jpg",
        "dataset/img1.jpg",
        "dataset/img2.jpg",
        "dataset/img6.jpg",
        "dataset/img6.jpg",
    ]

    for img_path in img_paths:
        for detector in detectors:
            results = DeepFace.analyze(
                img_path, actions=("gender",), detector_backend=detector, enforce_detection=False
            )
            for result in results:
                logger.debug(result)

                # validate keys
                assert "gender" in result.keys()
                assert "dominant_gender" in result.keys() and result["dominant_gender"] in [
                    "Man",
                    "Woman",
                ]

                # validate probabilities
                assert type(result) == dict
                if result["dominant_gender"] == "Man":
                    assert result["gender"]["Man"] > result["gender"]["Woman"]
                else:
                    assert result["gender"]["Man"] < result["gender"]["Woman"]

def test_analyze_for_batched_image():
    img = "dataset/img4.jpg"
    # Copy and combine the same image to create multiple faces
    img = cv2.imread(img)
    img = np.stack([img, img])
    assert len(img.shape) == 4 # Check dimension.
    assert img.shape[0] == 2 # Check batch size.

    demography_batch = DeepFace.analyze(img, silent=True)
    # 2 image in batch, so 2 demography objects.
    assert len(demography_batch) == 2 

    for demography_objs in demography_batch:
        assert len(demography_objs) == 1 # 1 face in each image
        for demography in demography_objs: # Iterate over faces
            assert type(demography) == dict # Check type
            assert demography["age"] > 20 and demography["age"] < 40
            assert demography["dominant_gender"] == "Woman"
    logger.info("✅ test analyze for multiple faces done")

def test_batch_detect_age_for_multiple_faces():
    # Load test image and resize to model input size
    img = cv2.resize(cv2.imread("dataset/img1.jpg"), (224, 224))
    imgs = [img, img]
    results = Age.ApparentAgeClient().predict(imgs)
    # Check there are two ages detected
    assert len(results) == 2
    # Check two faces ages are the same
    assert np.array_equal(results[0], results[1])
    logger.info("✅ test batch detect age for multiple faces done")

def test_batch_detect_emotion_for_multiple_faces():
    # Load test image and resize to model input size
    img = cv2.resize(cv2.imread("dataset/img1.jpg"), (224, 224))
    imgs = [img, img]
    results = Emotion.EmotionClient().predict(imgs)
    # Check there are two emotions detected
    assert len(results) == 2
    # Check two faces emotions are the same
    assert np.array_equal(results[0], results[1])
    logger.info("✅ test batch detect emotion for multiple faces done")

def test_batch_detect_gender_for_multiple_faces():
    # Load test image and resize to model input size
    img = cv2.resize(cv2.imread("dataset/img1.jpg"), (224, 224))
    imgs = [img, img]
    results = Gender.GenderClient().predict(imgs)
    # Check there are two genders detected
    assert len(results) == 2
    # Check two genders are the same
    assert np.array_equal(results[0], results[1])
    logger.info("✅ test batch detect gender for multiple faces done")

def test_batch_detect_race_for_multiple_faces():
    # Load test image and resize to model input size
    img = cv2.resize(cv2.imread("dataset/img1.jpg"), (224, 224))
    imgs = [img, img]
    results = Race.RaceClient().predict(imgs)
    # Check there are two races detected
    assert len(results) == 2
    # Check two races are the same
    assert np.array_equal(results[0], results[1])
    logger.info("✅ test batch detect race for multiple faces done")