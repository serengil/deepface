import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import cv2
from pathlib import Path
import gdown
import hashlib
import math
from PIL import Image
import copy
import base64

def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def distance(a, b):
	x1 = a[0]; y1 = a[1]
	x2 = b[0]; y2 = b[1]
	
	return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

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
			threshold = 0.64
	
	return threshold

def get_opencv_path():
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]
	
	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder

	face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
	eye_detector_path = path+"/data/haarcascade_eye.xml"
	
	if os.path.isfile(face_detector_path) != True:
		raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")
	
	return path+"/data/"

def detectFace(img, target_size=(224, 224), grayscale = False):
	
	#-----------------------
	
	exact_image = False
	if type(img).__module__ == np.__name__:
		exact_image = True
	
	base64_img = False
	if len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	#-----------------------
	
	opencv_path = get_opencv_path()
	face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
	eye_detector_path = opencv_path+"haarcascade_eye.xml"
	
	if os.path.isfile(face_detector_path) != True:
		raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")
	
	#--------------------------------
	
	face_detector = cv2.CascadeClassifier(face_detector_path)
	eye_detector = cv2.CascadeClassifier(eye_detector_path)
	
	if base64_img == True:
		img = loadBase64Img(img)
		
	elif exact_image != True: #image path passed as input
		
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")
		
		img = cv2.imread(img)
	
	img_raw = img.copy()
	
	#--------------------------------
	
	faces = face_detector.detectMultiScale(img, 1.3, 5)
	
	#print("found faces in ",image_path," is ",len(faces))
	
	if len(faces) > 0:
		x,y,w,h = faces[0]
		detected_face = img[int(y):int(y+h), int(x):int(x+w)]
		detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
		
		#---------------------------
		#face alignment
		
		eyes = eye_detector.detectMultiScale(detected_face_gray)
		
		if len(eyes) >= 2:
			#find the largest 2 eye
			base_eyes = eyes[:, 2]
			
			items = []
			for i in range(0, len(base_eyes)):
				item = (base_eyes[i], i)
				items.append(item)
			
			df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
			
			eyes = eyes[df.idx.values[0:2]]
			
			#-----------------------
			#decide left and right eye
			
			eye_1 = eyes[0]; eye_2 = eyes[1]
			
			if eye_1[0] < eye_2[0]:
				left_eye = eye_1
				right_eye = eye_2
			else:
				left_eye = eye_2
				right_eye = eye_1
			
			#-----------------------
			#find center of eyes
			
			left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
			left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
			
			right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
			right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
			
			#-----------------------
			#find rotation direction
				
			if left_eye_y > right_eye_y:
				point_3rd = (right_eye_x, left_eye_y)
				direction = -1 #rotate same direction to clock
			else:
				point_3rd = (left_eye_x, right_eye_y)
				direction = 1 #rotate inverse direction of clock
			
			#-----------------------
			#find length of triangle edges
			
			a = distance(left_eye_center, point_3rd)
			b = distance(right_eye_center, point_3rd)
			c = distance(right_eye_center, left_eye_center)
			
			#-----------------------
			#apply cosine rule
			
			cos_a = (b*b + c*c - a*a)/(2*b*c)
			angle = np.arccos(cos_a) #angle in radian
			angle = (angle * 180) / math.pi #radian to degree
			
			#-----------------------
			#rotate base image
			
			if direction == -1:
				angle = 90 - angle
			
			img = Image.fromarray(img_raw)
			img = np.array(img.rotate(direction * angle))
			
			#you recover the base image and face detection disappeared. apply again.
			faces = face_detector.detectMultiScale(img, 1.3, 5)
			if len(faces) > 0:
				x,y,w,h = faces[0]
				detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			
			#-----------------------
		
		#face alignment block end
		#---------------------------
		
		#face alignment block needs colorful images. that's why, converting to gray scale logic moved to here.
		if grayscale == True:
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
		
		detected_face = cv2.resize(detected_face, target_size)
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		#normalize input in [0, 1]
		img_pixels /= 255
		
		return img_pixels
		
	else:
		
		if exact_image == True:
			
			if grayscale == True:
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			img = cv2.resize(img, target_size)
			img_pixels = image.img_to_array(img)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255
			return img_pixels
		else:
			raise ValueError("Face could not be detected in ", img,". Please confirm that the picture is a face photo.")
