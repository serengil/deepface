from pathlib import Path
import gdown
import bz2
import os

def build_model():

	home = str(Path.home())

	import dlib #this requirement is not a must that's why imported here

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

	face_detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(home+"/.deepface/weights/shape_predictor_5_face_landmarks.dat")

	detector = {}
	detector["face_detector"] = face_detector
	detector["sp"] = sp
	return detector

def detect_face(detector, img, align = True):

	import dlib #this requirement is not a must that's why imported here

	home = str(Path.home())

	sp = detector["sp"]

	detected_face = None
	img_region = [0, 0, img.shape[0], img.shape[1]]

	face_detector = detector["face_detector"]
	detections = face_detector(img, 1)

	if len(detections) > 0:

		for idx, d in enumerate(detections):
			left = d.left(); right = d.right()
			top = d.top(); bottom = d.bottom()
			detected_face = img[top:bottom, left:right]
			img_region = [left, top, right - left, bottom - top]
			break #get the first one

		if align:
			img_shape = sp(img, detections[0])
			detected_face = dlib.get_face_chip(img, img_shape, size = detected_face.shape[0])

	return detected_face, img_region
