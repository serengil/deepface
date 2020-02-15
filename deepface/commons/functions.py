import os
from pathlib import Path
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import cv2
from pathlib import Path
import gdown
import hashlib

def findFileHash(file):
	BLOCK_SIZE = 65536 # The size of each read from the file

	file_hash = hashlib.sha256() # Create the hash object, can use something other than `.sha256()` if you wish
	with open(file, 'rb') as f: # Open the file to read it's bytes
		fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above
		while len(fb) > 0: # While there is still data being read from the file
			file_hash.update(fb) # Update the hash
			fb = f.read(BLOCK_SIZE) # Read the next block from the file
	
	return file_hash.hexdigest()

def initializeFolder():
	
	home = str(Path.home())
	
	if not os.path.exists(home+"/.deepface"):
		os.mkdir(home+"/.deepface")
		print("Directory ",home,"/.deepface created")
	
	if not os.path.exists(home+"/.deepface/weights"):
		os.mkdir(home+"/.deepface/weights")
		print("Directory ",home,"/.deepface/weights created")
	
	#----------------------------------
	"""
	#avoid interrupted file download
	
	weight_hashes = [
		['age_model_weights.h5', '0aeff75734bfe794113756d2bfd0ac823d51e9422c8961125b570871d3c2b114']
		, ['facenet_weights.h5', '90659cc97bfda5999120f95d8e122f4d262cca11715a21e59ba024bcce816d5c']
		, ['facial_expression_model_weights.h5', 'e8e8851d3fa05c001b1c27fd8841dfe08d7f82bb786a53ad8776725b7a1e824c']
		, ['gender_model_weights.h5', '45513ce5678549112d25ab85b1926fb65986507d49c674a3d04b2ba70dba2eb5']
		, ['openface_weights.h5', '5b41897ec6dd762cee20575eee54ed4d719a78cb982b2080a87dc14887d88a7a']
		, ['race_model_single_batch.h5', 'eb22b28b1f6dfce65b64040af4e86003a5edccb169a1a338470dde270b6f5e54']
		, ['vgg_face_weights.h5', '759266b9614d0fd5d65b97bf716818b746cc77ab5944c7bffc937c6ba9455d8c']
	]
	
	for i in weight_hashes:
		
		weight_file = home+"/.deepface/weights/"+i[0]
		expected_hash = i[1]
		
		#check file exits
		if os.path.isfile(weight_file) == True:
			current_hash = findFileHash(weight_file)
			if current_hash != expected_hash:
				print("hash violated for ", i[0],". It's going to be removed.")
				os.remove(weight_file)
	"""
	#----------------------------------

def findThreshold(model_name, distance_metric):
	
	threshold = 0.40
	
	if model_name == 'VGG-Face':
		if distance_metric == 'cosine':
			threshold = 0.40
		elif distance_metric == 'euclidean':
			threshold = 0.55
		elif distance_metric == 'euclidean_l2':
			threshold = 0.75	
	
	elif model_name == 'OpenFace':
		if distance_metric == 'cosine':
			threshold = 0.10
		elif distance_metric == 'euclidean':
			threshold = 0.55
		elif distance_metric == 'euclidean_l2':
			threshold = 0.55
	
	elif model_name == 'Facenet':
		if distance_metric == 'cosine':
			threshold = 0.40
		elif distance_metric == 'euclidean':
			threshold = 10
		elif distance_metric == 'euclidean_l2':
			threshold = 0.80
	
	elif model_name == 'DeepFace':
		if distance_metric == 'cosine':
			threshold = 0.23
		elif distance_metric == 'euclidean':
			threshold = 64
		elif distance_metric == 'euclidean_l2':
			threshold = 0.69
	
	return threshold

def detectFace(image_path, target_size=(224, 224), grayscale = False):
	
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]
	
	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder

	detector_path = path+"/data/haarcascade_frontalface_default.xml"
	
	if os.path.isfile(detector_path) != True:
		raise ValueError("Confirm that opencv is installed on your environment! Expected path ",detector_path," violated.")
	
	#--------------------------------
	
	detector = cv2.CascadeClassifier(detector_path)
	
	if grayscale != True:
		img = cv2.imread(image_path)
	else: #gray scale
		img = cv2.imread(image_path, 0)
	
	faces = detector.detectMultiScale(img, 1.3, 5)
	
	#print("found faces in ",image_path," is ",len(faces))
	
	if len(faces) > 0:
		x,y,w,h = faces[0]
		detected_face = img[int(y):int(y+h), int(x):int(x+w)]
		detected_face = cv2.resize(detected_face, target_size)
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		if True:
			#normalize input in [0, 1]
			img_pixels /= 255 
		else:
			#normalize input in [-1, +1]
			img_pixels /= 127.5
			img_pixels -= 1
		
		return img_pixels
		
	else:
		raise ValueError("Face could not be detected in ", image_path,". Please confirm that the picture is a face photo.")
