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
