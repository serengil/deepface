import os
from pathlib import Path
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import cv2
from pathlib import Path
import gdown

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
