import warnings
warnings.filterwarnings("ignore")

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import time
import unittest

#-----------------------------------------

import tensorflow as tf

class deepface_unit_tests(unittest.TestCase):

    def test_deepface(self):
        tf_version = int(tf.__version__.split(".")[0])

        if tf_version == 2:
            import logging
            tf.get_logger().setLevel(logging.ERROR)

        print("Running unit tests for TF ", tf.__version__)

        from deepface import DeepFace
        from deepface.commons import functions
        from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
        from deepface.extendedmodels import Age, Gender, Race, Emotion

        print("-----------------------------------------")
        #-----------------------------------------

        print("DeepFace.detectFace test")
        #detectors = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
        detectors = ['opencv', 'ssd', 'mtcnn', 'retinaface']

        for detector in detectors:
            img = DeepFace.detectFace("dataset/img11.jpg", detector_backend = detector)
            print(detector," test is done")

        #import matplotlib.pyplot as plt
        #plt.imshow(img)
        #plt.show()

        #-----------------------------------------
        print("-----------------------------------------")

        img_path = "dataset/img1.jpg"
        embedding = DeepFace.represent(img_path)
        print("Function returned ", len(embedding), "dimensional vector")

        model_name = "VGG-Face"
        model = DeepFace.build_model(model_name)
        print(model_name," is built")
        embedding = DeepFace.represent(img_path, model = model)
        print("Represent function returned ", len(embedding), "dimensional vector")

        #-----------------------------------------

        dataset = [
            ['dataset/img1.jpg', 'dataset/img2.jpg', True],
            ['dataset/img1.jpg', 'dataset/img6.jpg', True]
        ]

        print("-----------------------------------------")

        print("Face detectors test")

        print("retinaface detector")
        res = DeepFace.verify(dataset, detector_backend = 'retinaface')
        print(res)

        print("ssd detector")
        res = DeepFace.verify(dataset, detector_backend = 'ssd')
        print(res)

        print("opencv detector")
        res = DeepFace.verify(dataset, detector_backend = 'opencv')
        print(res)

        if False:
            print("dlib detector")
            res = DeepFace.verify(dataset, detector_backend = 'dlib')
            print(res)

        print("mtcnn detector")
        res = DeepFace.verify(dataset, detector_backend = 'mtcnn')
        print(res)

        print("-----------------------------------------")

        print("Single find function test")

        df = DeepFace.find(img_path = "dataset/img1.jpg", db_path = "dataset"
            #, model_name = 'Dlib'
        )
        print(df.head())

        print("-----------------------------------------")

        print("Pre-built model for single find function test")

        #model_name = "VGG-Face"
        #model = DeepFace.build_model(model_name)
        #print(model_name," is built")

        df = DeepFace.find(img_path = "dataset/img1.jpg", db_path = "dataset"
                            , model_name = model_name, model = model
        )
        print(df.head())

        print("-----------------------------------------")

        print("Bulk find function tests")

        dfs = DeepFace.find(img_path = ["dataset/img1.jpg", "dataset/img2.jpg"], db_path = "dataset"
            #, model_name = 'Dlib'
        )
        print(dfs[0].head())
        print(dfs[1].head())

        print("-----------------------------------------")

        print("Bulk verification tests")

        resp_obj = DeepFace.verify(dataset)
        print(resp_obj)
        print(resp_obj["pair_1"]["verified"] == True)
        print(resp_obj["pair_2"]["verified"] == True)

        print("-----------------------------------------")

        print("Bulk facial analysis tests")

        dataset = [
            'dataset/img1.jpg',
            'dataset/img2.jpg',
            'dataset/img5.jpg',
            'dataset/img6.jpg'
        ]

        resp_obj = DeepFace.analyze(dataset)
        print(resp_obj["instance_1"]["age"]," years old ", resp_obj["instance_1"]["dominant_emotion"], " ",resp_obj["instance_1"]["gender"])
        print(resp_obj["instance_2"]["age"]," years old ", resp_obj["instance_2"]["dominant_emotion"], " ",resp_obj["instance_2"]["gender"])
        print(resp_obj["instance_3"]["age"]," years old ", resp_obj["instance_3"]["dominant_emotion"], " ",resp_obj["instance_3"]["gender"])
        print(resp_obj["instance_4"]["age"]," years old ", resp_obj["instance_4"]["dominant_emotion"], " ",resp_obj["instance_4"]["gender"])

        print("-----------------------------------------")

        print("Facial analysis test. Passing nothing as an action")

        img = "dataset/img4.jpg"
        demography = DeepFace.analyze(img)
        print(demography)

        print("-----------------------------------------")

        print("Facial analysis test. Passing all to the action")
        demography = DeepFace.analyze(img, ['age', 'gender', 'race', 'emotion'])

        print("Demography:")
        print(demography)

        #check response is a valid json
        print("Age: ", demography["age"])
        print("Gender: ", demography["gender"])
        print("Race: ", demography["dominant_race"])
        print("Emotion: ", demography["dominant_emotion"])

        print("-----------------------------------------")

        print("Facial analysis test 2. Remove some actions and check they are not computed")
        demography = DeepFace.analyze(img, ['age', 'gender'])

        print("Age: ", demography.get("age"))
        print("Gender: ", demography.get("gender"))
        print("Race: ", demography.get("dominant_race"))
        print("Emotion: ", demography.get("dominant_emotion"))


        print("-----------------------------------------")

        print("Face recognition tests")

        dataset = [
            ['dataset/img1.jpg', 'dataset/img2.jpg', True],
            ['dataset/img5.jpg', 'dataset/img6.jpg', True],
            ['dataset/img6.jpg', 'dataset/img7.jpg', True],
            ['dataset/img8.jpg', 'dataset/img9.jpg', True],
            ['dataset/img1.jpg', 'dataset/img11.jpg', True],
            ['dataset/img2.jpg', 'dataset/img11.jpg', True],

            ['dataset/img1.jpg', 'dataset/img3.jpg', False],
            ['dataset/img2.jpg', 'dataset/img3.jpg', False],
            ['dataset/img6.jpg', 'dataset/img8.jpg', False],
            ['dataset/img6.jpg', 'dataset/img9.jpg', False],
        ]

        #models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace']
        models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace', 'SFace'] #those are robust models
        metrics = ['cosine', 'euclidean', 'euclidean_l2']

        passed_tests = 0; test_cases = 0

        for model in models:
            #prebuilt_model = DeepFace.build_model(model)
            #print(model," is built")
            for metric in metrics:
                for instance in dataset:
                    img1 = instance[0]
                    img2 = instance[1]
                    result = instance[2]

                    resp_obj = DeepFace.verify(img1, img2
                                , model_name = model
                                #, model = prebuilt_model
                                , distance_metric = metric)

                    prediction = resp_obj["verified"]
                    distance = round(resp_obj["distance"], 2)
                    threshold = resp_obj["threshold"]

                    test_result_label = "failed"
                    if prediction == result:
                        passed_tests = passed_tests + 1
                        test_result_label = "passed"

                    if prediction == True:
                        classified_label = "verified"
                    else:
                        classified_label = "unverified"

                    test_cases = test_cases + 1

                    print(img1.split("/")[-1], "-", img2.split("/")[-1], classified_label, "as same person based on", model,"and",metric,". Distance:",distance,", Threshold:", threshold,"(",test_result_label,")")

                print("--------------------------")

        #-----------------------------------------

        print("Passed unit tests: ",passed_tests," / ",test_cases)

        min_score = 70

        accuracy = 100 * passed_tests / test_cases
        accuracy = round(accuracy, 2)

        if accuracy >= min_score:
            print("Unit tests are completed successfully. Score: ",accuracy,"%")
        else:
            raise ValueError("Unit test score does not satisfy the minimum required accuracy. Minimum expected score is ", min_score,"% but this got ",accuracy,"%")

        #-----------------------------------
        #-----------------------------------

        print("Analyze function with passing pre-trained model")

        emotion_model = DeepFace.build_model("Emotion")
        age_model = DeepFace.build_model("Age")
        gender_model = DeepFace.build_model("Gender")
        race_model = DeepFace.build_model("Race")

        facial_attribute_models = {}
        facial_attribute_models["emotion"] = emotion_model
        facial_attribute_models["age"] = age_model
        facial_attribute_models["gender"] = gender_model
        facial_attribute_models["race"] = race_model

        resp_obj = DeepFace.analyze("dataset/img1.jpg", models=facial_attribute_models)
        print(resp_obj)

        #-----------------------------------
        print("--------------------------")

        if False:
            print("Ensemble for find function")
            df = DeepFace.find(img_path = "dataset/img1.jpg", db_path = "dataset", model_name = "Ensemble")
            print(df.head())

        #-----------------------------------
        print("--------------------------")

        if False:
            print("Ensemble for verify function")
            resp_obj = DeepFace.verify(dataset, model_name = "Ensemble")

            for i in range(0, len(dataset)):
                item = resp_obj['pair_%s' % (i+1)]
                verified = item["verified"]
                score = item["score"]
                print(verified)

        #-----------------------------------
        print("--------------------------")

        if False:

            print("Pre-trained ensemble method - find")

            from deepface import DeepFace
            from deepface.basemodels import Boosting

            model = Boosting.loadModel()
            df = DeepFace.find("dataset/img1.jpg", db_path = "dataset", model_name = 'Ensemble', model = model, enforce_detection=False)

            print(df)

        #-----------------------------------
        print("--------------------------")

        if False:
            print("Pre-trained ensemble method - verify")
            res = DeepFace.verify(dataset, model_name = "Ensemble", model = model)
            print(res)

        #-----------------------------------
        print("--------------------------")

        import cv2

        print("Passing numpy array to analyze function")

        img = cv2.imread("dataset/img1.jpg")
        resp_obj = DeepFace.analyze(img)
        print(resp_obj)

        print("--------------------------")

        print("Passing numpy array to verify function")

        img1 = cv2.imread("dataset/img1.jpg")
        img2 = cv2.imread("dataset/img2.jpg")

        res = DeepFace.verify(img1, img2)
        print(res)

        print("--------------------------")

        print("Passing numpy array to find function")

        img1 = cv2.imread("dataset/img1.jpg")

        df = DeepFace.find(img1, db_path = "dataset")

        print(df.head())

        print("--------------------------")

        self.assertEqual(accuracy >= min_score, False, "A problem on the deepface installation.")

unittest.main(exit=False)