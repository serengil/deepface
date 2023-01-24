import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#------------------------------

from flask import Flask, jsonify, request, make_response

import argparse
import uuid
import json
import time
from tqdm import tqdm

#------------------------------

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

#------------------------------

if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

#------------------------------

from deepface import DeepFace

#------------------------------

app = Flask(__name__)

#------------------------------

if tf_version == 1:
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

	if tf_version == 1:
		with graph.as_default():
			resp_obj = analyzeWrapper(req, trx_id)
	elif tf_version == 2:
		resp_obj = analyzeWrapper(req, trx_id)

	#---------------------------

	toc = time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj, 200

def analyzeWrapper(req, trx_id = 0):
	resp_obj = {}

	instances = []
	if "img" in list(req.keys()):
		raw_content = req["img"] #list

		for item in raw_content: #item is in type of dict
			instances.append(item)

	if len(instances) == 0:
		return {'success': False, 'error': 'you must pass at least one img object in your request'}

	print("Analyzing ", len(instances)," instances")

	#---------------------------

	detector_backend = 'opencv'
	actions= ['emotion', 'age', 'gender', 'race']
	align = True
	enforce_detection = True

	if "actions" in list(req.keys()):
		actions = req["actions"]

	if "detector_backend" in list(req.keys()):
		detector_backend = req["detector_backend"]
	
	if "align" in list(req.keys()):
		align = req["align"]
	
	if "enforce_detection" in list(req.keys()):
		enforce_detection = req["enforce_detection"]

	#---------------------------

	try:
		resp_obj["demographies"] = {}
		for idx, instance in enumerate(instances):

			demographies = DeepFace.analyze(img_path = instance, 
										detector_backend = detector_backend, 
										actions = actions, align = align, 
										enforce_detection = enforce_detection)
			resp_obj["demographies"][f"img_{idx+1}"] = demographies

	except Exception as err:
		print("Exception: ", str(err))
		return jsonify({'success': False, 'error': str(err)}), 205

	#---------------
	return resp_obj

@app.route('/verify', methods=['POST'])
def verify():

	global graph

	tic = time.time()
	req = request.get_json()
	trx_id = uuid.uuid4()

	resp_obj = jsonify({'success': False})

	if tf_version == 1:
		with graph.as_default():
			resp_obj = verifyWrapper(req, trx_id)
	elif tf_version == 2:
		resp_obj = verifyWrapper(req, trx_id)

	#--------------------------

	toc =  time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj, 200

def verifyWrapper(req, trx_id = 0):

	resp_obj = {}

	model_name = "VGG-Face"; distance_metric = "cosine"; detector_backend = "opencv"
	align = True; enforce_detection = True
	if "model_name" in list(req.keys()):
		model_name = req["model_name"]
	if "distance_metric" in list(req.keys()):
		distance_metric = req["distance_metric"]
	if "detector_backend" in list(req.keys()):
		detector_backend = req["detector_backend"]
	if "align" in list(req.keys()):
		align = req["align"]
	if "enforce_detection" in list(req.keys()):
		enforce_detection = req["enforce_detection"]
	#----------------------
	try:
		if "img" in list(req.keys()):
			raw_content = req["img"] #list

			if len(raw_content) == 0:
				return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205
			
			print("Input request of ", trx_id, " has ",len(raw_content)," pairs to verify")

			results = []
			for idx, item in enumerate(raw_content): #item is in type of dict
				img1 = item["img1"]; img2 = item["img2"]

				validate_img1 = False
				if len(img1) > 11 and img1[0:11] == "data:image/":
					validate_img1 = True

				validate_img2 = False
				if len(img2) > 11 and img2[0:11] == "data:image/":
					validate_img2 = True

				if validate_img1 != True or validate_img2 != True:
					return jsonify({'success': False, 'error': 'you must pass both img1 and img2 as base64 encoded string'}), 205
				
				result = DeepFace.verify(img1_path=img1, 
								img2_path=img2, 
								model_name=model_name, 
								detector_backend=detector_backend, 
								distance_metric=distance_metric,
								align=align,
								enforce_detection=enforce_detection,
								)
				results.append(result)
				
			resp_obj[f"pairs"] = results
	except Exception as err:
		resp_obj = jsonify({'success': False, 'error': str(err)}), 205		

	return resp_obj

@app.route('/represent', methods=['POST'])
def represent():

	global graph

	tic = time.time()
	req = request.get_json()
	trx_id = uuid.uuid4()

	resp_obj = jsonify({'success': False})

	if tf_version == 1:
		with graph.as_default():
			resp_obj = representWrapper(req, trx_id)
	elif tf_version == 2:
		resp_obj = representWrapper(req, trx_id)

	#--------------------------

	toc =  time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj, 200

def representWrapper(req, trx_id = 0):

	resp_obj = jsonify({'success': False})

	#-------------------------------------
	#find out model

	model_name = "VGG-Face"; detector_backend = 'opencv'

	if "model_name" in list(req.keys()):
		model_name = req["model_name"]

	if "detector_backend" in list(req.keys()):
		detector_backend = req["detector_backend"]

	#-------------------------------------
	#retrieve images from request

	img = ""
	if "img" in list(req.keys()):
		img = req["img"] #list
		#print("img: ", img)

	validate_img = False
	if len(img) > 11 and img[0:11] == "data:image/":
		validate_img = True

	if validate_img != True:
		print("invalid image passed!")
		return jsonify({'success': False, 'error': 'you must pass img as base64 encoded string'}), 205

	#-------------------------------------
	#call represent function from the interface

	try:

		embedding_objs = DeepFace.represent(img
			, model_name = model_name
			, detector_backend = detector_backend
		)

	except Exception as err:
		print("Exception: ",str(err))
		resp_obj = jsonify({'success': False, 'error': str(err)}), 205

	#-------------------------------------

	#print("embedding is ", len(embedding)," dimensional vector")
	resp_obj = {}
	faces = []
	for embedding_obj in embedding_objs:
		face = {}
		face["embedding"] = embedding_obj["embedding"]
		face["facial_area"] = embedding_obj["facial_area"]
		face["model_name"] = model_name
		face["detector_backend"] = detector_backend
		faces.append(face)

	resp_obj["embeddings"] = faces

	#-------------------------------------

	return resp_obj

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--port',
		type=int,
		default=5000,
		help='Port of serving api')
	args = parser.parse_args()
	app.run(host='0.0.0.0', port=args.port)
