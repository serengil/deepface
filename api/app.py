from flask import Flask, jsonify, request, make_response

import uuid
import json
import time

from deepface import DeepFace

#------------------------------

app = Flask(__name__)

#------------------------------
#Service API Interface

@app.route('/')
def index():
	return '<h1>Hello, world!</h1>'

@app.route('/analyze', methods=['POST'])
def analyze():

	req = request.get_json()
	trx_id = uuid.uuid4()

	#---------------------------

	tic = time.time()

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

	resp_obj = DeepFace.analyze(instances, actions=actions)
	
	#---------------------------

	toc = time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj

@app.route('/verify', methods=['POST'])

def verify():

	req = request.get_json()
	trx_id = uuid.uuid4()

	tic = time.time()
	
	#-------------------------
	
	model_name = "VGG-Face"; distance_metric = "cosine"
	if "model_name" in list(req.keys()):
		model_name = req["model_name"]
	if "distance_metric" in list(req.keys()):
		distance_metric = req["distance_metric"]
	
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
	resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric)

	toc =  time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	#--------------------------
	
	return resp_obj, 200

if __name__ == '__main__':
	
	app.run()