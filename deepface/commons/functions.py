import os
import numpy as np
import pandas as pd
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
import bz2
from deepface.commons import distance
from mtcnn import MTCNN #0.1.0

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
	from tensorflow.keras.applications.imagenet_utils import preprocess_input
	from tensorflow.keras.preprocessing import image

#--------------------------------------------------

def initialize_input(img1_path, img2_path = None):

	if type(img1_path) == list:
		bulkProcess = True
		img_list = img1_path.copy()
	else:
		bulkProcess = False

		if (
			(type(img2_path) == str and img2_path != None) #exact image path, base64 image
			or (isinstance(img2_path, np.ndarray) and img2_path.any()) #numpy array
		):
			img_list = [[img1_path, img2_path]]
		else: #analyze function passes just img1_path
			img_list = [img1_path]

	return img_list, bulkProcess

def initialize_detector(detector_backend):

	global face_detector

	home = str(Path.home())

	#eye detector is common for opencv and ssd
	if detector_backend == 'opencv' or detector_backend == 'ssd':
		opencv_path = get_opencv_path()
		eye_detector_path = opencv_path+"haarcascade_eye.xml"

		if os.path.isfile(eye_detector_path) != True:
			raise ValueError("Confirm that opencv is installed on your environment! Expected path ",eye_detector_path," violated.")

		global eye_detector
		eye_detector = cv2.CascadeClassifier(eye_detector_path)

	#------------------------------
	#face detectors
	if detector_backend == 'opencv':
		opencv_path = get_opencv_path()
		face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"

		if os.path.isfile(face_detector_path) != True:
			raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")

		face_detector = cv2.CascadeClassifier(face_detector_path)

	elif detector_backend == 'ssd':

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

		face_detector = cv2.dnn.readNetFromCaffe(
			home+"/.deepface/weights/deploy.prototxt",
			home+"/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"
		)

	elif detector_backend == 'dlib':
		import dlib #this is not a must library within deepface. that's why, I didn't put this import to a global level. version: 19.20.0

		global sp

		face_detector = dlib.get_frontal_face_detector()

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

		sp = dlib.shape_predictor(home+"/.deepface/weights/shape_predictor_5_face_landmarks.dat")

	elif detector_backend == 'mtcnn':
		face_detector = MTCNN()

	elif detector_backend == 'retinaface':
		from retinaface import RetinaFace
		face_detector = RetinaFace.build_model()

def initializeFolder():

	home = str(Path.home())

	if not os.path.exists(home+"/.deepface"):
		os.mkdir(home+"/.deepface")
		print("Directory ",home,"/.deepface created")

	if not os.path.exists(home+"/.deepface/weights"):
		os.mkdir(home+"/.deepface/weights")
		print("Directory ",home,"/.deepface/weights created")

def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

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

	img_region = [0, 0, img.shape[0], img.shape[1]]

	#if functions.preproces_face is called directly, then face_detector global variable might not been initialized.
	if not "face_detector" in globals():
		initialize_detector(detector_backend = detector_backend)

	if detector_backend == 'opencv':

		faces = []

		try:
			faces = face_detector.detectMultiScale(img, 1.3, 5)
		except:
			pass

		if len(faces) > 0:
			x,y,w,h = faces[0] #focus on the 1st face found in the image
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			return detected_face, [x, y, w, h]

		else: #if no face detected

			if enforce_detection != True:
				return img, img_region

			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

	elif detector_backend == 'ssd':

		ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

		target_size = (300, 300)

		base_img = img.copy() #we will restore base_img to img later

		original_size = img.shape

		img = cv2.resize(img, target_size)

		aspect_ratio_x = (original_size[1] / target_size[1])
		aspect_ratio_y = (original_size[0] / target_size[0])

		imageBlob = cv2.dnn.blobFromImage(image = img)

		face_detector.setInput(imageBlob)
		detections = face_detector.forward()

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

			return detected_face, [int(left*aspect_ratio_x), int(top*aspect_ratio_y), int(right*aspect_ratio_x) - int(left*aspect_ratio_x), int(bottom*aspect_ratio_y) - int(top*aspect_ratio_y)]

		else: #if no face detected

			if enforce_detection != True:
				img = base_img.copy()
				return img, img_region

			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

	elif detector_backend == 'dlib':

		detections = face_detector(img, 1)

		if len(detections) > 0:

			for idx, d in enumerate(detections):
				left = d.left(); right = d.right()
				top = d.top(); bottom = d.bottom()

				detected_face = img[top:bottom, left:right]

				return detected_face, [left, top, right - left, bottom - top]

		else: #if no face detected

			if enforce_detection != True:
				return img, img_region

			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

	elif detector_backend == 'mtcnn':

		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #mtcnn expects RGB but OpenCV read BGR
		detections = face_detector.detect_faces(img_rgb)

		if len(detections) > 0:
			detection = detections[0]
			x, y, w, h = detection["box"]
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			return detected_face, [x, y, w, h]

		else: #if no face detected
			if not enforce_detection:
				return img, img_region

			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

	elif detector_backend == 'retinaface':
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR

		from retinaface import RetinaFace
		faces = RetinaFace.extract_faces(img_rgb, align = True)

		if len(faces) > 0:
			face = faces[0]
			return face, img_region

		else: #if no face detected
			if not enforce_detection:
				return img, img_region
			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")


	else:
		detectors = ['opencv', 'ssd', 'dlib', 'mtcnn']
		raise ValueError("Valid backends are ", detectors," but you passed ", detector_backend)

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

		import dlib #this is not a must dependency in deepface

		detections = face_detector(img, 1)

		if len(detections) > 0:
			detected_face = detections[0]
			img_shape = sp(img, detected_face)
			img = dlib.get_face_chip(img, img_shape, size = img.shape[0])

		return img #return img anyway

	elif detector_backend == 'mtcnn':

		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #mtcnn expects RGB but OpenCV read BGR
		detections = face_detector.detect_faces(img_rgb)

		if len(detections) > 0:
			detection = detections[0]

			keypoints = detection["keypoints"]
			left_eye = keypoints["left_eye"]
			right_eye = keypoints["right_eye"]

			img = alignment_procedure(img, left_eye, right_eye)

		return img #return img anyway
	elif detector_backend == 'retinaface':
		#we used extract_faces function of retinaface. it applies alignment anyway.
		return img #return img anyway

def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False):

	#img_path = copy.copy(img)

	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	img = load_image(img)
	base_img = img.copy()

	img, region = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection)

	#--------------------------

	if img.shape[0] > 0 and img.shape[1] > 0:
		img = align_face(img = img, detector_backend = detector_backend)
	else:

		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: #restore base image
			img = base_img.copy()

	#--------------------------

	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img = cv2.resize(img, target_size)
	img_pixels = image.img_to_array(img)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	if return_region == True:
		return img_pixels, region
	else:
		return img_pixels

def find_input_shape(model):

	#face recognition models have different size of inputs
	#my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

	input_shape = model.layers[0].input_shape

	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]

	if type(input_shape) == list: #issue 197: some people got array here instead of tuple
		input_shape = tuple(input_shape)

	return input_shape
