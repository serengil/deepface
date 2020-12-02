from deepface import DeepFace
from tqdm import tqdm
import os
from os import path
from pathlib import Path
import numpy as np
import lightgbm as lgb #lightgbm==2.3.1

from deepface.commons import functions, distance as dst

def loadModel():
	
	model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']

	model = {}
	
	model_pbar = tqdm(range(0, 4), desc='Face recognition models')
	
	for index in model_pbar:
		
		model_name = model_names[index]
		
		model_pbar.set_description("Loading %s" % (model_name))
		model[model_name] = DeepFace.build_model(model_name)
	
	return model

def validate_model(model):
	#validate model dictionary because it might be passed from input as pre-trained
	found_models = []
	for key, value in model.items():
		found_models.append(key)
	
	if ('VGG-Face' in found_models) and ('Facenet' in found_models) and ('OpenFace' in found_models) and ('DeepFace' in found_models):
		#print("Ensemble learning will be applied for ", found_models," models")
		valid = True
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
