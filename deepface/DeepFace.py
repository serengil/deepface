import matplotlib.pyplot as plt
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

#from basemodels import VGGFace, OpenFace, Facenet, Age, Gender, Race, Emotion
#from commons import functions, distance as dst

from deepface.basemodels import VGGFace, OpenFace, Facenet, Age, Gender, Race, Emotion
from deepface.commons import functions, distance as dst

def verify(img1_path, img2_path
	, model_name ='VGG-Face', distance_metric = 'cosine'):
	
	tic = time.time()
	
	if os.path.isfile(img1_path) != True:
		raise ValueError("Confirm that ",img1_path," exists")
	
	if os.path.isfile(img2_path) != True:
		raise ValueError("Confirm that ",img2_path," exists")
		
	#-------------------------
	
	#print("Face verification will be applied on ",model_name," model and ",distance_metric," metric")
	
	functions.validateInputs(model_name, distance_metric)
	
	#-------------------------
	
	#tuned thresholds for model and metric pair
	threshold = functions.findThreshold(model_name, distance_metric)
	
	#-------------------------
	
	if model_name == 'VGG-Face':
		model = VGGFace.loadModel()
		input_shape = (224, 224)	
	
	elif model_name == 'OpenFace':
		model = OpenFace.loadModel()
		input_shape = (96, 96)
	
	elif model_name == 'Facenet':
		model = Facenet.loadModel()
		input_shape = (160, 160)
	
	#-------------------------
	#crop face
	
	img1 = functions.detectFace(img1_path, input_shape)
	img2 = functions.detectFace(img2_path, input_shape)
	
	#-------------------------
	#TO-DO: Apply face alignment here. Experiments show that aligment increases accuracy 1%.
	
	#-------------------------
	#find embeddings
	
	img1_representation = model.predict(img1)[0,:]
	img2_representation = model.predict(img2)[0,:]
	
	#-------------------------
	#find distances between embeddings
	
	if distance_metric == 'cosine':
		distance = dst.findCosineDistance(img1_representation, img2_representation)
	elif distance_metric == 'euclidean':
		distance = dst.findEuclideanDistance(img1_representation, img2_representation)
	elif distance_metric == 'euclidean_l2':
		distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
	
	#-------------------------
	#decision
	
	if distance <= threshold:
		identified =  True
	else:
		identified =  False
	
	#-------------------------
	
	plot = False
	
	if plot:
		label = "Distance is "+str(round(distance, 2))
		
		fig = plt.figure()
		fig.add_subplot(1,2, 1)
		plt.imshow(img1[0][:, :, ::-1])
		plt.xticks([]); plt.yticks([])
		fig.add_subplot(1,2, 2)
		plt.imshow(img2[0][:, :, ::-1])
		plt.xticks([]); plt.yticks([])
		fig.suptitle(label, fontsize=17)
		plt.show(block=True)
	
	#-------------------------
	
	toc = time.time()
	
	#print("identification lasts ",toc-tic," seconds")
	
	#Return a tuple. First item is the identification result based on tuned threshold.
	#Second item is the threshold. You might want to customize this threshold to identify faces.
	return (identified, distance, threshold)

def analyze(img_path, actions= []):
	
	resp_obj = "{\n   "
	
	#if a specific target is not passed, then find them all
	if len(actions) == 0:
		actions= ['emotion', 'age', 'gender', 'race']
	
	print("Actions to do: ", actions)
	
	img = functions.detectFace(img_path, (224, 224))
	
	#TO-DO: do this in parallel
	
	pbar = tqdm(range(0,len(actions)), desc='Finding actions')
	
	action_idx = 0
	#for action in actions:
	for index in pbar:
		action = actions[index]
		pbar.set_description("Action: %s" % (action))
		
		if action_idx > 0:
			resp_obj += "\n   , "
		
		if action == 'emotion':
			emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
			img = functions.detectFace(img_path, (48, 48), True)
			
			model = Emotion.loadModel()
			emotion_predictions = model.predict(img)[0,:]
			
			sum_of_predictions = emotion_predictions.sum()
			
			emotion_obj = "\"emotion\": {"
			for i in range(0, len(emotion_labels)):
				emotion_label = emotion_labels[i]
				emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
				
				if i > 0: emotion_obj += ", "
				
				emotion_obj += "\n      "
				emotion_obj += "\"%s\": %s" % (emotion_label, emotion_prediction)
			
			emotion_obj += "\n   }"
			
			emotion_obj += "\n   , \"dominant_emotion\": \"%s\"" % (emotion_labels[np.argmax(emotion_predictions)])
			
			resp_obj += emotion_obj
			
		elif action == 'age':
			#print("age prediction")
			model = Age.loadModel()
			age_predictions = model.predict(img)[0,:]
			apparent_age = Age.findApparentAge(age_predictions)
			
			resp_obj += "\"age\": %s" % (apparent_age)
			
		elif action == 'gender':
			#print("gender prediction")
			
			model = Gender.loadModel()
			gender_prediction = model.predict(img)[0,:]
			
			if np.argmax(gender_prediction) == 0:
				gender = "Woman"
			elif np.argmax(gender_prediction) == 1:
				gender = "Man"
			
			resp_obj += "\"gender\": \"%s\"" % (gender)
			
		elif action == 'race':
			model = Race.loadModel()
			race_predictions = model.predict(img)[0,:]
			race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
			
			sum_of_predictions = race_predictions.sum()
			
			race_obj = "\"race\": {"
			for i in range(0, len(race_labels)):
				race_label = race_labels[i]
				race_prediction = 100 * race_predictions[i] / sum_of_predictions
				
				if i > 0: race_obj += ", "
				
				race_obj += "\n      "
				race_obj += "\"%s\": %s" % (race_label, race_prediction)
			
			race_obj += "\n   }"
			race_obj += "\n   , \"dominant_race\": \"%s\"" % (race_labels[np.argmax(race_predictions)])
			
			resp_obj += race_obj
		
		action_idx = action_idx + 1
	
	resp_obj += "\n}"
	return resp_obj
#---------------------------

functions.initializeFolder()

#---------------------------