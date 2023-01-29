import warnings
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace

print("-----------------------------------------")

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf_major_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_major_version == 2:
    import logging

    tf.get_logger().setLevel(logging.ERROR)

print("Running unit tests for TF ", tf.__version__)

print("-----------------------------------------")

expected_coverage = 97
num_cases = 0
succeed_cases = 0


def evaluate(condition):

    global num_cases, succeed_cases

    if condition == True:
        succeed_cases += 1

    num_cases += 1


# ------------------------------------------------

detectors = ["opencv", "mtcnn"]
models = ["VGG-Face", "Facenet", "ArcFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

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

print("-----------------------------------------")


def test_cases():

    print("Enforce detection test")
    black_img = np.zeros([224, 224, 3])

    # enforce detection on for represent
    try:
        DeepFace.represent(img_path=black_img)
        exception_thrown = False
    except:
        exception_thrown = True

    assert exception_thrown is True

    # -------------------------------------------

    # enforce detection off for represent
    try:
        objs = DeepFace.represent(img_path=black_img, enforce_detection=False)
        exception_thrown = False

        # validate response of represent function
        assert isinstance(objs, list)
        assert len(objs) > 0
        assert isinstance(objs[0], dict)
        assert "embedding" in objs[0].keys()
        assert "facial_area" in objs[0].keys()
        assert isinstance(objs[0]["facial_area"], dict)
        assert "x" in objs[0]["facial_area"].keys()
        assert "y" in objs[0]["facial_area"].keys()
        assert "w" in objs[0]["facial_area"].keys()
        assert "h" in objs[0]["facial_area"].keys()
        assert isinstance(objs[0]["embedding"], list)
        assert len(objs[0]["embedding"]) == 2622  # embedding of VGG-Face
    except Exception as err:
        print(str(err))
        exception_thrown = True

    assert exception_thrown is False

    # -------------------------------------------
    # enforce detection on for verify
    try:
        obj = DeepFace.verify(img1_path=black_img, img2_path=black_img)
        exception_thrown = False
    except:
        exception_thrown = True

    assert exception_thrown is True
    # -------------------------------------------
    # enforce detection off for verify

    try:
        obj = DeepFace.verify(img1_path=black_img, img2_path=black_img, enforce_detection=False)
        assert isinstance(obj, dict)
        exception_thrown = False
    except:
        exception_thrown = True

    assert exception_thrown is False
    # -------------------------------------------

    print("-----------------------------------------")

    print("Extract faces test")

    for detector in detectors:
        img_objs = DeepFace.extract_faces(img_path="dataset/img11.jpg", detector_backend=detector)
        for img_obj in img_objs:
            assert "face" in img_obj.keys()
            assert "facial_area" in img_obj.keys()
            assert isinstance(img_obj["facial_area"], dict)
            assert "x" in img_obj["facial_area"].keys()
            assert "y" in img_obj["facial_area"].keys()
            assert "w" in img_obj["facial_area"].keys()
            assert "h" in img_obj["facial_area"].keys()
            assert "confidence" in img_obj.keys()

            img = img_obj["face"]
            evaluate(img.shape[0] > 0 and img.shape[1] > 0)
            print(detector, " test is done")

    print("-----------------------------------------")

    img_path = "dataset/img1.jpg"
    embedding_objs = DeepFace.represent(img_path)
    for embedding_obj in embedding_objs:
        embedding = embedding_obj["embedding"]
        print("Function returned ", len(embedding), "dimensional vector")
        evaluate(len(embedding) == 2622)

    print("-----------------------------------------")

    print("Different face detectors on verification test")

    for detector in detectors:
        print(detector + " detector")
        res = DeepFace.verify(dataset[0][0], dataset[0][1], detector_backend=detector)

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

        print(res)
        evaluate(res["verified"] == dataset[0][2])

    print("-----------------------------------------")

    print("Find function test")

    dfs = DeepFace.find(img_path="dataset/img1.jpg", db_path="dataset")
    for df in dfs:
        assert isinstance(df, pd.DataFrame)
        print(df.head())
        evaluate(df.shape[0] > 0)

    print("-----------------------------------------")

    print("Facial analysis test. Passing nothing as an action")

    img = "dataset/img4.jpg"
    demography_objs = DeepFace.analyze(img)
    for demography in demography_objs:
        print(demography)
        evaluate(demography["age"] > 20 and demography["age"] < 40)
        evaluate(demography["dominant_gender"] == "Woman")

    print("-----------------------------------------")

    print("Facial analysis test. Passing all to the action")
    demography_objs = DeepFace.analyze(img, ["age", "gender", "race", "emotion"])

    for demography in demography_objs:
        # print(f"Demography: {demography}")
        # check response is a valid json
        print("Age: ", demography["age"])
        print("Gender: ", demography["dominant_gender"])
        print("Race: ", demography["dominant_race"])
        print("Emotion: ", demography["dominant_emotion"])

        evaluate(demography.get("age") is not None)
        evaluate(demography.get("dominant_gender") is not None)
        evaluate(demography.get("dominant_race") is not None)
        evaluate(demography.get("dominant_emotion") is not None)

    print("-----------------------------------------")

    print("Facial analysis test 2. Remove some actions and check they are not computed")
    demography_objs = DeepFace.analyze(img, ["age", "gender"])

    for demography in demography_objs:
        print("Age: ", demography.get("age"))
        print("Gender: ", demography.get("dominant_gender"))
        print("Race: ", demography.get("dominant_race"))
        print("Emotion: ", demography.get("dominant_emotion"))

        evaluate(demography.get("age") is not None)
        evaluate(demography.get("dominant_gender") is not None)
        evaluate(demography.get("dominant_race") is None)
        evaluate(demography.get("dominant_emotion") is None)

    print("-----------------------------------------")

    print("Facial recognition tests")

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

                passed = prediction == result

                evaluate(passed)

                if passed:
                    test_result_label = "passed"
                else:
                    test_result_label = "failed"

                if prediction == True:
                    classified_label = "verified"
                else:
                    classified_label = "unverified"

                print(
                    img1.split("/", maxsplit=1)[-1],
                    "-",
                    img2.split("/", maxsplit=1)[-1],
                    classified_label,
                    "as same person based on",
                    model,
                    "and",
                    metric,
                    ". Distance:",
                    distance,
                    ", Threshold:",
                    threshold,
                    "(",
                    test_result_label,
                    ")",
                )

            print("--------------------------")

    # -----------------------------------------

    print("Passing numpy array to analyze function")

    img = cv2.imread("dataset/img1.jpg")
    resp_objs = DeepFace.analyze(img)

    for resp_obj in resp_objs:
        print(resp_obj)
        evaluate(resp_obj["age"] > 20 and resp_obj["age"] < 40)
        evaluate(resp_obj["gender"] == "Woman")

    print("--------------------------")

    print("Passing numpy array to verify function")

    img1 = cv2.imread("dataset/img1.jpg")
    img2 = cv2.imread("dataset/img2.jpg")

    res = DeepFace.verify(img1, img2)
    print(res)
    evaluate(res["verified"] == True)

    print("--------------------------")

    print("Passing numpy array to find function")

    img1 = cv2.imread("dataset/img1.jpg")

    dfs = DeepFace.find(img1, db_path="dataset")

    for df in dfs:
        print(df.head())
        evaluate(df.shape[0] > 0)

    print("--------------------------")

    print("non-binary gender tests")

    # interface validation - no need to call evaluate here

    for img1_path, _, _ in dataset:
        for detector in detectors:
            results = DeepFace.analyze(
                img1_path, actions=("gender",), detector_backend=detector, enforce_detection=False
            )

            for result in results:
                print(result)

                assert "gender" in result.keys()
                assert "dominant_gender" in result.keys() and result["dominant_gender"] in [
                    "Man",
                    "Woman",
                ]

                if result["dominant_gender"] == "Man":
                    assert result["gender"]["Man"] > result["gender"]["Woman"]
                else:
                    assert result["gender"]["Man"] < result["gender"]["Woman"]


# ---------------------------------------------

test_cases()

print("num of test cases run: " + str(num_cases))
print("succeeded test cases: " + str(succeed_cases))

test_score = (100 * succeed_cases) / num_cases

print("test coverage: " + str(test_score))

if test_score > expected_coverage:
    print("well done! min required test coverage is satisfied")
else:
    print("min required test coverage is NOT satisfied")

assert test_score > expected_coverage
