from deepface import DeepFace
import json

#-----------------------------------------

print("Facial analysis tests")

img = "dataset/img4.jpg"
demography = DeepFace.analyze(img, ['age', 'gender', 'race', 'emotion'])

print("Demography:")
print(demography)
demography = json.loads(demography)

#check response is a valid json
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Race: ", demography["dominant_race"])
print("Emotion: ", demography["dominant_emotion"])

print("-----------------------------------------")

print("Face recognition tests")

dataset = [
	['dataset/img1.jpg', 'dataset/img2.jpg', True],
	['dataset/img1.jpg', 'dataset/img3.jpg', False],
	['dataset/img2.jpg', 'dataset/img3.jpg', False],
]

models = ['VGG-Face', 'Facenet', 'OpenFace']
metrics = ['cosine', 'euclidean', 'euclidean_l2']

passed_tests = 0; test_cases = 0

for model in models:
	for metric in metrics:
		for instance in dataset:
			img1 = instance[0]
			img2 = instance[1]
			result = instance[2]
			
			idx = DeepFace.verify(img1, img2, model_name = model, distance_metric = metric)
			
			test_result_label = "failed"
			if idx[0] == result:
				passed_tests = passed_tests + 1
				test_result_label = "passed"
			
			if idx[0] == True:
				classified_label = "verified"
			else:
				classified_label = "unverified"
			
			test_cases = test_cases + 1
			
			print(img1, " and ", img2," are ", classified_label, " as same person based on ", model," model and ",metric," distance metric. Distance: ",round(idx[1], 2),", Required Threshold: ", idx[2]," (",test_result_label,")")
		
		print("--------------------------")

#-----------------------------------------

print("Passed unit tests: ",passed_tests," / ",test_cases)

accuracy = 100 * passed_tests / test_cases
accuracy = round(accuracy, 2)

if accuracy > 80:
	print("Unit tests are completed successfully. Score: ",accuracy,"%")
else:
	raise ValueError("Unit test score does not satisfy the minimum required accuracy. Minimum expected score is 80% but this got ",accuracy,"%")
