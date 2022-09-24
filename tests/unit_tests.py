import warnings
import os
import tensorflow as tf
import cv2
import sys
from deepface import DeepFace

print("-----------------------------------------", file=sys.stderr)

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_major_version = int(tf.__version__.split(".")[0])

if tf_major_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

print("Running unit tests for TF ", tf.__version__, file=sys.stderr)

print("-----------------------------------------", file=sys.stderr)

expected_coverage = 97
num_cases = 0; succeed_cases = 0

def evaluate(condition):

	global num_cases, succeed_cases

	if condition is True:
		succeed_cases += 1
	
	num_cases += 1

# ------------------------------------------------

detectors = ['opencv', 'mtcnn', 'retinaface']
models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace', 'SFace']
metrics = ['cosine', 'euclidean', 'euclidean_l2']

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

print("-----------------------------------------", file=sys.stderr)

def test_cases():

	print("DeepFace.detectFace test", file=sys.stderr)

	for detector in detectors:
		img = DeepFace.detectFace("dataset/img11.jpg", detector_backend = detector)
		evaluate(img.shape[0] > 0 and img.shape[1] > 0)
		print(detector," test is done", file=sys.stderr)

	print("-----------------------------------------", file=sys.stderr)

	img_path = "dataset/img1.jpg"
	embedding = DeepFace.represent(img_path)
	print("Function returned ", len(embedding), "dimensional vector", file=sys.stderr)
	evaluate(len(embedding) > 0)

	print("-----------------------------------------", file=sys.stderr)

	print("Face detectors test", file=sys.stderr)

	for detector in detectors:
		print(detector + " detector", file=sys.stderr)
		res = DeepFace.verify(dataset[0][0], dataset[0][1], detector_backend = detector)
		print(res, file=sys.stderr)
		assert res["verified"] == dataset[0][2]

	print("-----------------------------------------", file=sys.stderr)

	print("Find function test", file=sys.stderr)

	df = DeepFace.find(img_path = "dataset/img1.jpg", db_path = "dataset")
	print(df.head(), file=sys.stderr)
	evaluate(df.shape[0] > 0)

	print("-----------------------------------------", file=sys.stderr)

	print("Facial analysis test. Passing nothing as an action", file=sys.stderr)

	img = "dataset/img4.jpg"
	demography = DeepFace.analyze(img)
	print(demography, file=sys.stderr)

	evaluate(demography["age"] > 20 and demography["age"] < 40)
	evaluate(demography["dominant_gender"] == "Woman")

	print("-----------------------------------------", file=sys.stderr)

	print("Facial analysis test. Passing all to the action", file=sys.stderr)
	demography = DeepFace.analyze(img, ['age', 'gender', 'race', 'emotion'])

	print("Demography:", file=sys.stderr)
	print(demography, file=sys.stderr)

	#check response is a valid json
	print("Age: ", demography["age"], file=sys.stderr)
	print("Gender: ", demography["dominant_gender"], file=sys.stderr)
	print("Race: ", demography["dominant_race"], file=sys.stderr)
	print("Emotion: ", demography["dominant_emotion"], file=sys.stderr)

	evaluate(demography.get("age") is not None)
	evaluate(demography.get("dominant_gender") is not None)
	evaluate(demography.get("dominant_race") is not None)
	evaluate(demography.get("dominant_emotion") is not None)

	print("-----------------------------------------", file=sys.stderr)

	print("Facial analysis test 2. Remove some actions and check they are not computed", file=sys.stderr)
	demography = DeepFace.analyze(img, ['age', 'gender'])

	print("Age: ", demography.get("age"), file=sys.stderr)
	print("Gender: ", demography.get("dominant_gender"), file=sys.stderr)
	print("Race: ", demography.get("dominant_race"), file=sys.stderr)
	print("Emotion: ", demography.get("dominant_emotion"), file=sys.stderr)

	evaluate(demography.get("age") is not None)
	evaluate(demography.get("dominant_gender") is not None)
	evaluate(demography.get("dominant_race") is None)
	evaluate(demography.get("dominant_emotion") is None)

	print("-----------------------------------------", file=sys.stderr)

	print("Facial recognition tests", file=sys.stderr)

	for model in models:
		for metric in metrics:
			for instance in dataset:
				img1 = instance[0]
				img2 = instance[1]
				result = instance[2]

				resp_obj = DeepFace.verify(img1, img2
							, model_name = model
							, distance_metric = metric)

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

				print(img1.split("/")[-1], "-", img2.split("/")[-1], classified_label, "as same person based on", model,"and",metric,". Distance:",distance,", Threshold:", threshold,"(",test_result_label,")", file=sys.stderr)

			print("--------------------------", file=sys.stderr)

	# -----------------------------------------
	
	print("Passing numpy array to analyze function", file=sys.stderr)

	img = cv2.imread("dataset/img1.jpg")
	resp_obj = DeepFace.analyze(img)
	print(resp_obj, file=sys.stderr)

	evaluate(resp_obj["age"] > 20 and resp_obj["age"] < 40)
	evaluate(resp_obj["gender"] == "Woman")

	print("--------------------------", file=sys.stderr)

	print("Passing numpy array to verify function", file=sys.stderr)

	img1 = cv2.imread("dataset/img1.jpg")
	img2 = cv2.imread("dataset/img2.jpg")

	res = DeepFace.verify(img1, img2)
	print(res, file=sys.stderr)

	evaluate(res["verified"] == True)

	print("--------------------------", file=sys.stderr)

	print("Passing numpy array to find function", file=sys.stderr)

	img1 = cv2.imread("dataset/img1.jpg")

	df = DeepFace.find(img1, db_path = "dataset")

	print(df.head(), file=sys.stderr)

	evaluate(df.shape[0] > 0)

	print("--------------------------", file=sys.stderr)

	print("non-binary gender tests", file=sys.stderr)

	#interface validation - no need to call evaluate here

	for img1_path, img2_path, verified in dataset:
		for detector in detectors:
			result = DeepFace.analyze(img1_path, actions=('gender',), detector_backend=detector, enforce_detection=False)

			print(result, file=sys.stderr)

			assert 'gender' in result.keys()
			assert 'dominant_gender' in result.keys() and result["dominant_gender"] in ["Man", "Woman"]

			if result["dominant_gender"] == "Man":
				assert result["gender"]["Man"] > result["gender"]["Woman"]
			else:
				assert result["gender"]["Man"] < result["gender"]["Woman"]

# ---------------------------------------------

test_cases()

print("num of test cases run: " + str(num_cases), file=sys.stderr)
print("succeeded test cases: " + str(succeed_cases), file=sys.stderr)

test_score = (100 * succeed_cases) / num_cases

print("test coverage: " + str(test_score), file=sys.stderr)

if test_score > expected_coverage:
	print("well done! min required test coverage is satisfied", file=sys.stderr)
else:
	print("min required test coverage is NOT satisfied", file=sys.stderr)

assert test_score > expected_coverage
