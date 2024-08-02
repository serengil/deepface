# 3rd party dependencies
import cv2

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


detectors = ["opencv", "mtcnn"]


def test_standard_analyze():
    img = "dataset/img4.jpg"
    demography_objs = DeepFace.analyze(img, silent=True)
    for demography in demography_objs:
        logger.debug(demography)
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
                if result["dominant_gender"] == "Man":
                    assert result["gender"]["Man"] > result["gender"]["Woman"]
                else:
                    assert result["gender"]["Man"] < result["gender"]["Woman"]
