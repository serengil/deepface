from deepface import DeepFace
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#-----------------------------------------

print("Bulk tests")

print("-----------------------------------------")

print("Large scale face recognition")

df = DeepFace.find(img_path = "dataset/img1.jpg", db_path = "dataset")
print(df.head())

print("-----------------------------------------")

print("Ensemble for find function")
df = DeepFace.find(img_path = "dataset/img1.jpg", db_path = "dataset", model_name = "Ensemble")
print(df.head())

print("-----------------------------------------")

print("Bulk face recognition tests")

dataset = [
	['dataset/img1.jpg', 'dataset/img2.jpg', True],
	['dataset/img5.jpg', 'dataset/img6.jpg', True]
]

resp_obj = DeepFace.verify(dataset)
print(resp_obj["pair_1"]["verified"] == True)
print(resp_obj["pair_2"]["verified"] == True)

print("-----------------------------------------")

print("Ensemble learning bulk")
resp_obj = DeepFace.verify(dataset, model_name = "Ensemble")

for i in range(0, len(dataset)):
	item = resp_obj['pair_%s' % (i+1)]
	verified = item["verified"]
	score = item["score"]
	print(verified)

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

#-----------------------------------------

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

models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID']
metrics = ['cosine', 'euclidean', 'euclidean_l2']

passed_tests = 0; test_cases = 0

for model in models:
	for metric in metrics:
		for instance in dataset:
			img1 = instance[0]
			img2 = instance[1]
			result = instance[2]
			
			resp_obj = DeepFace.verify(img1, img2, model_name = model, distance_metric = metric)
			prediction = resp_obj["verified"]
			distance = round(resp_obj["distance"], 2)
			required_threshold = resp_obj["max_threshold_to_verify"]
			
			test_result_label = "failed"
			if prediction == result:
				passed_tests = passed_tests + 1
				test_result_label = "passed"
			
			if prediction == True:
				classified_label = "verified"
			else:
				classified_label = "unverified"
			
			test_cases = test_cases + 1
			
			print(img1, " and ", img2," are ", classified_label, " as same person based on ", model," model and ",metric," distance metric. Distance: ",distance,", Required Threshold: ", required_threshold," (",test_result_label,")")
		
		print("--------------------------")

#-----------------------------------------

print("Passed unit tests: ",passed_tests," / ",test_cases)

accuracy = 100 * passed_tests / test_cases
accuracy = round(accuracy, 2)

if accuracy > 75:
	print("Unit tests are completed successfully. Score: ",accuracy,"%")
else:
	raise ValueError("Unit test score does not satisfy the minimum required accuracy. Minimum expected score is 80% but this got ",accuracy,"%")

#-----------------------------------

# api tests - already built models will be passed to the functions

from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace

#-----------------------------------

vggface_model = VGGFace.loadModel()
resp_obj = DeepFace.verify("dataset/img1.jpg", "dataset/img2.jpg", model_name = "VGG-Face", model = vggface_model)
print(resp_obj)

#-----------------------------------

from deepface.extendedmodels import Age, Gender, Race, Emotion

emotion_model = Emotion.loadModel()
age_model = Age.loadModel()
gender_model = Gender.loadModel()
race_model = Race.loadModel()

facial_attribute_models = {}
facial_attribute_models["emotion"] = emotion_model
facial_attribute_models["age"] = age_model
facial_attribute_models["gender"] = gender_model
facial_attribute_models["race"] = race_model

resp_obj = DeepFace.analyze("dataset/img1.jpg", models=facial_attribute_models)

