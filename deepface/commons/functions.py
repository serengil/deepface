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
import multiprocessing
import subprocess
import tensorflow as tf
import keras
import bz2
from deepface.commons import distance

def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def initializeFolder():
	
	home = str(Path.home())
	
	if not os.path.exists(home+"/.deepface"):
		os.mkdir(home+"/.deepface")
		print("Directory ",home,"/.deepface created")
	
	if not os.path.exists(home+"/.deepface/weights"):
		os.mkdir(home+"/.deepface/weights")
		print("Directory ",home,"/.deepface/weights created")
	
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
	
	elif model_name == 'DeepID':
		if distance_metric == 'cosine':
			threshold = 0.015
		elif distance_metric == 'euclidean':
			threshold = 45
		elif distance_metric == 'euclidean_l2':
			threshold = 0.17
	
	elif model_name == 'Dlib':
		if distance_metric == 'cosine':
			threshold = 0.07
		elif distance_metric == 'euclidean':
			threshold = 0.60
		elif distance_metric == 'euclidean_l2':
			threshold = 0.60
	
	return threshold

def get_opencv_path():
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]
	
	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder
	
	return path+"/data/"

def load_image(img):
	
	exact_image = False
	if type(img).__module__ == np.__name__:
		exact_image = True
	
	base64_img = False
	if len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True
	
	#---------------------------
	
	if base64_img == True:
		img = loadBase64Img(img)
		
	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")
		
		img = cv2.imread(img)
	
	return img
	
def detect_face(img, detector_backend = 'opencv', grayscale = False, enforce_detection = True):
	
	home = str(Path.home())
	
	if detector_backend == 'opencv':
	
		#get opencv configuration up first
		opencv_path = get_opencv_path()
		face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
	
		if os.path.isfile(face_detector_path) != True:
			raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")
		
		face_detector = cv2.CascadeClassifier(face_detector_path)
	
		#--------------------------
		
		faces = []
		
		try: 
			faces = face_detector.detectMultiScale(img, 1.3, 5)
		except:
			pass
		
		if len(faces) > 0:
			x,y,w,h = faces[0] #focus on the 1st face found in the image
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			return detected_face
		
		else: #if no face detected
	
			if enforce_detection != True:			
				return img
	
			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

	elif detector_backend == 'ssd':
		
		#---------------------------
		#check required ssd model exists in the home/.deepface/weights folder
		
		#model structure
		if os.path.isfile(home+'/.deepface/weights/deploy.prototxt') != True:
			
			print("deploy.prototxt will be downloaded...")
			
			url = "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt"
			
			output = home+'/.deepface/weights/deploy.prototxt'
			
			gdown.download(url, output, quiet=False)
			
		
		#pre-trained weights
		if os.path.isfile(home+'/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel') != True:
			
			print("res10_300x300_ssd_iter_140000.caffemodel will be downloaded...")
			
			url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
			
			output = home+'/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel'
			
			gdown.download(url, output, quiet=False)
			
		#---------------------------
		
		ssd_detector = cv2.dnn.readNetFromCaffe(
			home+"/.deepface/weights/deploy.prototxt", 
			home+"/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"
		)
		
		ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
		
		target_size = (300, 300)
		
		base_img = img.copy() #we will restore base_img to img later
		
		original_size = img.shape
		
		img = cv2.resize(img, target_size)
		
		aspect_ratio_x = (original_size[1] / target_size[1])
		aspect_ratio_y = (original_size[0] / target_size[0])
		
		imageBlob = cv2.dnn.blobFromImage(image = img)
		
		ssd_detector.setInput(imageBlob)
		detections = ssd_detector.forward()
		
		detections_df = pd.DataFrame(detections[0][0], columns = ssd_labels)
		
		detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
		detections_df = detections_df[detections_df['confidence'] >= 0.90]
		
		detections_df['left'] = (detections_df['left'] * 300).astype(int)
		detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
		detections_df['right'] = (detections_df['right'] * 300).astype(int)
		detections_df['top'] = (detections_df['top'] * 300).astype(int)
		
		if detections_df.shape[0] > 0:
			
			#TODO: sort detections_df
			
			#get the first face in the image	
			instance = detections_df.iloc[0]
			
			left = instance["left"]
			right = instance["right"]
			bottom = instance["bottom"]
			top = instance["top"]
			
			detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
			
			return detected_face
			
		else: #if no face detected
	
			if enforce_detection != True:
				img = base_img.copy()
				return img
	
			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")
	
	elif detector_backend == 'dlib':
		import dlib #this is not a must library within deepface. that's why, I didn't put this import to a global level. version: 19.20.0
		
		detector = dlib.get_frontal_face_detector()
		
		detections = detector(img, 1)
		
		if len(detections) > 0:
			
			for idx, d in enumerate(detections):
				left = d.left(); right = d.right()
				top = d.top(); bottom = d.bottom()
				
				detected_face = img[top:bottom, left:right]
				
				return detected_face
			
		else: #if no face detected
	
			if enforce_detection != True:			
				return img
	
			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.") 
		
	elif detector_backend == 'mtcnn':
		from mtcnn import MTCNN #0.1.0
		mtcnn_detector = MTCNN()
		
		detections = mtcnn_detector.detect_faces(img)
		
		if len(detections) > 0:
			detection = detections[0]
			x, y, w, h = detection["box"]
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			return detected_face
		
		else: #if no face detected
			if enforce_detection != True:			
				return img
	
			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")
	
	else:
		detectors = ['opencv', 'ssd', 'dlib', 'mtcnn']
		raise ValueError("Valid backends are ", detectors," but you passed ", detector_backend)
	
	return 0

def alignment_procedure(img, left_eye, right_eye):
		
	#this function aligns given face in img based on left and right eye coordinates
	
	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye
	
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
	
	a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))
	
	#-----------------------
	
	#apply cosine rule
			
	if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
		
		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a) #angle in radian
		angle = (angle * 180) / math.pi #radian to degree
		
		#-----------------------
		#rotate base image
		
		if direction == -1:
			angle = 90 - angle
		
		img = Image.fromarray(img)
		img = np.array(img.rotate(direction * angle))
	
	#-----------------------
	
	return img #return img anyway
	
def align_face(img, detector_backend = 'opencv'):
	
	home = str(Path.home())
	
	if (detector_backend == 'opencv') or (detector_backend == 'ssd'):
	
		opencv_path = get_opencv_path()
		eye_detector_path = opencv_path+"haarcascade_eye.xml"
		eye_detector = cv2.CascadeClassifier(eye_detector_path)
		
		detected_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #eye detector expects gray scale image
		
		eyes = eye_detector.detectMultiScale(detected_face_gray)
		
		if len(eyes) >= 2:
			
			#find the largest 2 eye
			
			base_eyes = eyes[:, 2]
			
			items = []
			for i in range(0, len(base_eyes)):
				item = (base_eyes[i], i)
				items.append(item)
			
			df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
			
			eyes = eyes[df.idx.values[0:2]] #eyes variable stores the largest 2 eye
			
			#-----------------------
			#decide left and right eye
			
			eye_1 = eyes[0]; eye_2 = eyes[1]
			
			if eye_1[0] < eye_2[0]:
				left_eye = eye_1; right_eye = eye_2
			else:
				left_eye = eye_2; right_eye = eye_1
			
			#-----------------------
			#find center of eyes
			
			left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
			right_eye = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
			
			img = alignment_procedure(img, left_eye, right_eye)
			
		return img #return img anyway
	
	elif detector_backend == 'dlib':
	
		#check required file exists in the home/.deepface/weights folder
		
		if os.path.isfile(home+'/.deepface/weights/shape_predictor_5_face_landmarks.dat') != True:
			
			print("shape_predictor_5_face_landmarks.dat.bz2 is going to be downloaded") 
			
			url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
			output = home+'/.deepface/weights/'+url.split("/")[-1]
			
			gdown.download(url, output, quiet=False)
			
			zipfile = bz2.BZ2File(output)
			data = zipfile.read()
			newfilepath = output[:-4] #discard .bz2 extension
			open(newfilepath, 'wb').write(data)
		
		#------------------------------
		
		import dlib #this is not a must dependency in deepface
		
		detector = dlib.get_frontal_face_detector()
		sp = dlib.shape_predictor(home+"/.deepface/weights/shape_predictor_5_face_landmarks.dat")
		
		detections = detector(img, 1)
		
		if len(detections) > 0:
			detected_face = detections[0]
			img_shape = sp(img, detected_face)
			img = dlib.get_face_chip(img, img_shape, size = img.shape[0])
			
		return img #return img anyway
	
	elif detector_backend == 'mtcnn':
		
		from mtcnn import MTCNN #0.1.0
		mtcnn_detector = MTCNN()
		detections = mtcnn_detector.detect_faces(img)
		
		if len(detections) > 0:
			detection = detections[0]
			
			keypoints = detection["keypoints"]
			left_eye = keypoints["left_eye"]
			right_eye = keypoints["right_eye"]
			
			img = alignment_procedure(img, left_eye, right_eye)
				
		return img #return img anyway
	
def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv'):
	
	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	img = load_image(img)
	
	#--------------------------
	
	#we will align base image instead of detected face not have black pixels
	
	img = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection)
	img = align_face(img = img, detector_backend = detector_backend)
	
	#--------------------------
	
	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
	img = cv2.resize(img, target_size)
	img_pixels = image.img_to_array(img)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]
	
	return img_pixels
	
def allocateMemory():
	
	#find allocated memories
	gpu_indexes = []
	memory_usage_percentages = []; available_memories = []; total_memories = []; utilizations = []
	power_usages = []; power_capacities = []
	
	try:
		result = subprocess.check_output(['nvidia-smi'])

		dashboard = result.decode("utf-8").split("=|")

		dashboard = dashboard[1].split("\n")
		
		gpu_idx = 0
		for line in dashboard:
			if ("MiB" in line):
				power_info = line.split("|")[1]
				power_capacity = int(power_info.split("/")[-1].replace("W", ""))
				power_usage = int((power_info.split("/")[-2]).strip().split(" ")[-1].replace("W", ""))
				
				power_usages.append(power_usage)
				power_capacities.append(power_capacity)
				
				#----------------------------
				
				memory_info = line.split("|")[2].replace("MiB","").split("/")
				utilization_info = int(line.split("|")[3].split("%")[0])
				
				allocated = int(memory_info[0])
				total_memory = int(memory_info[1])
				available_memory = total_memory - allocated
				
				total_memories.append(total_memory)
				available_memories.append(available_memory)
				memory_usage_percentages.append(round(100*int(allocated)/int(total_memory), 4))
				utilizations.append(utilization_info)
				gpu_indexes.append(gpu_idx)
				
				gpu_idx = gpu_idx + 1
		
		gpu_count = gpu_idx * 1
				
	except Exception as err:
		gpu_count = 0
		#print(str(err))
		
	#------------------------------
	
	df = pd.DataFrame(gpu_indexes, columns = ["gpu_index"])
	df["total_memories_in_mb"] = total_memories
	df["available_memories_in_mb"] = available_memories
	df["memory_usage_percentage"] = memory_usage_percentages
	df["utilizations"] = utilizations
	df["power_usages_in_watts"] = power_usages
	df["power_capacities_in_watts"] = power_capacities
	
	df = df.sort_values(by = ["available_memories_in_mb"], ascending = False).reset_index(drop = True)
	
	#------------------------------
	
	required_memory = 10000 #All deepface models require 9016 MiB
	
	if df.shape[0] > 0: #has gpu
		if df.iloc[0].available_memories_in_mb > required_memory:
			my_gpu = str(int(df.iloc[0].gpu_index))
			os.environ["CUDA_VISIBLE_DEVICES"] = my_gpu
			
			#------------------------------
			#tf allocates all memory by default
			#this block avoids greedy approach
			
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			session = tf.Session(config=config)
			keras.backend.set_session(session)
			
			print("DeepFace will run on GPU (gpu_", my_gpu,")")
		else:
			#this case has gpu but no enough memory to allocate
			os.environ["CUDA_VISIBLE_DEVICES"] = "" #run it on cpu
			print("Even though the system has GPUs, there is no enough space in memory to allocate.")
			print("DeepFace will run on CPU")
	else:
		print("DeepFace will run on CPU")
