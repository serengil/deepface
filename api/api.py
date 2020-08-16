from flask import Flask, jsonify, request, make_response

import argparse
import uuid
import json
import time
from tqdm import tqdm

import tensorflow as tf

from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
from deepface.basemodels.DlibResNet import DlibResNet
from deepface.extendedmodels import Age, Gender, Race, Emotion

#import DeepFace
#from basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
#from extendedmodels import Age, Gender, Race, Emotion

#------------------------------

app = Flask(__name__)

#------------------------------

tic = time.time()

print("Loading Face Recognition Models...")

pbar = tqdm(range(0,6), desc='Loading Face Recognition Models...')

for index in pbar:
	if index == 0:
		pbar.set_description("Loading VGG-Face")
		vggface_model = VGGFace.loadModel()
	elif index == 1:
		pbar.set_description("Loading OpenFace")
		openface_model = OpenFace.loadModel()
	elif index == 2:
		pbar.set_description("Loading Google FaceNet")
		facenet_model = Facenet.loadModel()
	elif index == 3:
		pbar.set_description("Loading Facebook DeepFace")
		deepface_model = FbDeepFace.loadModel()
	elif index == 4:
		pbar.set_description("Loading DeepID DeepFace")
		deepid_model = DeepID.loadModel()
	elif index == 5:
		pbar.set_description("Loading Dlib ResNet DeepFace")
		dlib_model = DlibResNet()

toc = time.time()

print("Face recognition models are built in ", toc-tic," seconds")

#------------------------------

tic = time.time()

print("Loading Facial Attribute Analysis Models...")

pbar = tqdm(range(0,4), desc='Loading Facial Attribute Analysis Models...')

for index in pbar:
	if index == 0:
		pbar.set_description("Loading emotion analysis model")
		emotion_model = Emotion.loadModel()
	elif index == 1:
		pbar.set_description("Loading age prediction model")
		age_model = Age.loadModel()
	elif index == 2:
		pbar.set_description("Loading gender prediction model")
		gender_model = Gender.loadModel()
	elif index == 3:
		pbar.set_description("Loading race prediction model")
		race_model = Race.loadModel()

toc = time.time()

facial_attribute_models = {}
facial_attribute_models["emotion"] = emotion_model
facial_attribute_models["age"] = age_model
facial_attribute_models["gender"] = gender_model
facial_attribute_models["race"] = race_model

print("Facial attribute analysis models are built in ", toc-tic," seconds")

#------------------------------

graph = tf.get_default_graph()

#------------------------------
#Service API Interface

@app.route('/')
def index():
	return '<h1>Hello, world!</h1>'

@app.route('/analyze', methods=['POST'])
def analyze():
	
	global graph
	
	tic = time.time()
	req = request.get_json()
	trx_id = uuid.uuid4()

	#---------------------------
	
	resp_obj = jsonify({'success': False})
	with graph.as_default():
		instances = []
		if "img" in list(req.keys()):
			raw_content = req["img"] #list

			for item in raw_content: #item is in type of dict
				instances.append(item)
		
		if len(instances) == 0:
			return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205
		
		print("Analyzing ", len(instances)," instances")

		#---------------------------

		actions= ['emotion', 'age', 'gender', 'race']
		if "actions" in list(req.keys()):
			actions = req["actions"]
		
		#---------------------------

		#resp_obj = DeepFace.analyze(instances, actions=actions)
		resp_obj = DeepFace.analyze(instances, actions=actions, models=facial_attribute_models)
		
		#---------------------------

	toc = time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj, 200

@app.route('/verify', methods=['POST'])

def verify():
	
	global graph
	
	tic = time.time()
	req = request.get_json()
	trx_id = uuid.uuid4()
	
	resp_obj = jsonify({'success': False})
	
	with graph.as_default():
		
		model_name = "VGG-Face"; distance_metric = "cosine"
		if "model_name" in list(req.keys()):
			model_name = req["model_name"]
		if "distance_metric" in list(req.keys()):
			distance_metric = req["distance_metric"]
		
		#----------------------
		
		instances = []
		if "img" in list(req.keys()):
			raw_content = req["img"] #list

			for item in raw_content: #item is in type of dict
				instance = []
				img1 = item["img1"]; img2 = item["img2"]

				validate_img1 = False
				if len(img1) > 11 and img1[0:11] == "data:image/":
					validate_img1 = True
				
				validate_img2 = False
				if len(img2) > 11 and img2[0:11] == "data:image/":
					validate_img2 = True

				if validate_img1 != True or validate_img2 != True:
					return jsonify({'success': False, 'error': 'you must pass both img1 and img2 as base64 encoded string'}), 205

				instance.append(img1); instance.append(img2)
				instances.append(instance)
			
		#--------------------------

		if len(instances) == 0:
			return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205
		
		print("Input request of ", trx_id, " has ",len(instances)," pairs to verify")
		
		#--------------------------
		
		if model_name == "VGG-Face":
			resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = vggface_model)
		elif model_name == "Facenet":
			resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = facenet_model)
		elif model_name == "OpenFace":
			resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = openface_model)
		elif model_name == "DeepFace":
			resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = deepface_model)
		elif model_name == "DeepID":
			resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = deepid_model)
		elif model_name == "Dlib":
			resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = dlib_model)
		elif model_name == "Ensemble":
			models =  {}
			models["VGG-Face"] = vggface_model
			models["Facenet"] = facenet_model
			models["OpenFace"] = openface_model
			models["DeepFace"] = deepface_model
			
			resp_obj = DeepFace.verify(instances, model_name = model_name, model = models)
			
		else:
			return jsonify({'success': False, 'error': 'You must pass a valid model name. Available models are VGG-Face, Facenet, OpenFace, DeepFace but you passed %s' % (model_name)}), 205
		
	#--------------------------
	
	toc =  time.time()
	
	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic
	
	return resp_obj, 200


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--port',
		type=int,
		default=5000,
		help='Port of serving api')
	args = parser.parse_args()
	app.run(host='0.0.0.0', port=args.port)
