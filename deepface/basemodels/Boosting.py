from deepface import DeepFace
from tqdm import tqdm
import os
from os import path
from pathlib import Path
import numpy as np
import lightgbm as lgb #lightgbm==2.3.1

from deepface.commons import functions, distance as dst

def loadModel():
	model = {}
	
	model_pbar = tqdm(range(0, 4), desc='Face recognition models')
	
	for index in model_pbar:
		
		if index == 0:
			model_pbar.set_description("Loading VGG-Face")
			model["VGG-Face"] = DeepFace.build_model('VGG-Face')
		elif index == 1:
			model_pbar.set_description("Loading Google FaceNet")
			model["Facenet"] = DeepFace.build_model('Facenet')
		elif index == 2:
			model_pbar.set_description("Loading OpenFace")
			model["OpenFace"] = DeepFace.build_model('OpenFace')
		elif index == 3:
			model_pbar.set_description("Loading Facebook DeepFace")
			model["DeepFace"] = DeepFace.build_model('DeepFace')
	
	return model

def validate_model(model):
	#validate model dictionary because it might be passed from input as pre-trained
	found_models = []
	for key, value in model.items():
		found_models.append(key)
	
	if ('VGG-Face' in found_models) and ('Facenet' in found_models) and ('OpenFace' in found_models) and ('DeepFace' in found_models):
		print("Ensemble learning will be applied for ", found_models," models")
	else:
		raise ValueError("You would like to apply ensemble learning and pass pre-built models but models must contain [VGG-Face, Facenet, OpenFace, DeepFace] but you passed "+found_models)

def build_gbm():
	
	home = str(Path.home())
	
	if os.path.isfile(home+'/.deepface/weights/face-recognition-ensemble-model.txt') != True:
		print("face-recognition-ensemble-model.txt will be downloaded...")
		url = 'https://raw.githubusercontent.com/serengil/deepface/master/deepface/models/face-recognition-ensemble-model.txt'
		output = home+'/.deepface/weights/face-recognition-ensemble-model.txt'
		gdown.download(url, output, quiet=False)
		
	ensemble_model_path = home+'/.deepface/weights/face-recognition-ensemble-model.txt'
	
	deepface_ensemble = lgb.Booster(model_file = ensemble_model_path)
	
	return deepface_ensemble

def verify(model, img_list, bulkProcess, enforce_detection, detector_backend):
	print("Ensemble learning enabled")
	
	if model == None:
		model = loadModel()
	
	validate_model(model)
	
	#--------------------------
	
	model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
	metrics = ["cosine", "euclidean", "euclidean_l2"]
	
	resp_objects = []
	
	#--------------------------
	
	if model == None:
		model = loadModel()
	
	#--------------------------
	
	validate_model(model)
	
	#--------------------------
	
	pbar = tqdm(range(0,len(img_list)), desc='Verification')
	
	for index in pbar:
		instance = img_list[index]
		
		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]
			img2_path = instance[1]
			
			ensemble_features = []; ensemble_features_string = "["
			
			for i in  model_names:
				custom_model = model[i]
				
				input_shape = custom_model.layers[0].input_shape
				
				if type(input_shape) == list:
					input_shape = input_shape[0][1:3]
				else:
					input_shape = input_shape[1:3]
				
				#----------------------------------
				
				img1 = functions.preprocess_face(img = img1_path, target_size = input_shape
						, enforce_detection = enforce_detection
						, detector_backend = detector_backend)
				
				img2 = functions.preprocess_face(img = img2_path, target_size = input_shape
						, enforce_detection = enforce_detection
						, detector_backend = detector_backend)
				
				img1_representation = custom_model.predict(img1)[0,:]
				img2_representation = custom_model.predict(img2)[0,:]
				
				for j in metrics:
					if j == 'cosine':
						distance = dst.findCosineDistance(img1_representation, img2_representation)
					elif j == 'euclidean':
						distance = dst.findEuclideanDistance(img1_representation, img2_representation)
					elif j == 'euclidean_l2':
						distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
				
					#------------------------
					
					#this returns same with OpenFace - euclidean_l2
					if i == 'OpenFace' and j == 'euclidean':
						continue
					else:
						ensemble_features.append(distance)
						
						if len(ensemble_features) > 1:
							ensemble_features_string += ", "
						
						ensemble_features_string += str(distance)
						
			ensemble_features_string += "]"
			
			#-------------------------------
			
			deepface_ensemble = build_gbm()
				
			#-------------------------------
			
			prediction = deepface_ensemble.predict(np.expand_dims(np.array(ensemble_features), axis=0))[0]
			
			verified = np.argmax(prediction) == 1
			
			score = prediction[np.argmax(prediction)]
				
			#print("verified: ", verified,", score: ", score)
			
			resp_obj = {
				"verified": verified
				, "score": score
				, "distance": ensemble_features_string
				, "model": ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
				, "similarity_metric": ["cosine", "euclidean", "euclidean_l2"]
			}
			
			if bulkProcess == True:
				resp_objects.append(resp_obj)
			else:
				return resp_obj
				
	if bulkProcess == True:
		resp_obj = {}
		
		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["pair_%d" % (i+1)] = resp_item
			
		return resp_obj

	