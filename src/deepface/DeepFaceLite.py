from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
import time
import os
from os import path
from pathlib import Path
import gdown
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import cv2
from keras import backend as K
import keras
import tensorflow as tf
import pickle

from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functionsLite, realtime, distance as dst


class DeepFaceLite(object):

	def __init__(self):

		functionsLite.initializeFolder()

        # init models
		self.detector_backend = 'mtcnn'
		self.emotion_model = Emotion.loadModel()
		self.age_model = Age.loadModel()
		self.gender_model = Gender.loadModel()

		# TODO: init detector


	def analyze(self, img, enforce_detection = True, detector_backend = 'opencv'):

		# preprocess images
		processed = functionsLite.preprocess_face(img, enforce_detection=enforce_detection, detector_backend=detector_backend)
		imgs_224 = processed['processed']
		emotion_imgs = processed['gray']
		bbox_img = processed['bbox']
		# original_faces = processed['original']

		resp_objects = []

		# iterate through faces
		for i in range(len(imgs_224)):

			resp_obj = {}

			# --- emotion --- 
			emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

			emotion_predictions = self.emotion_model.predict(emotion_imgs[i])[0,:]

			sum_of_predictions = emotion_predictions.sum()

			all_emotions = []

			for j in range(0, len(emotion_labels)):
				emotion_label = emotion_labels[j]
				emotion_prediction = 100 * emotion_predictions[j] / sum_of_predictions
				# all_emotions[emotion_label] = '{:.4f}'.format(emotion_prediction)
				max_score = 100 * np.max(emotion_predictions) / sum_of_predictions

				all_emotions.append('{}: {:.4f}'.format(emotion_label, emotion_prediction))

			emotion = {
				'all': all_emotions,
				'dominant': emotion_labels[np.argmax(emotion_predictions)],
				'dominant_score': '{:.4f}'.format(max_score)
			}

			print(emotion)

			# --- age --- 
			age_predictions = self.age_model.predict(imgs_224[i])[0,:]
			apparent_age = Age.findApparentAge(age_predictions)

			# --- gender --- 
			gender_prediction = self.gender_model.predict(imgs_224[i])[0,:]

			if np.argmax(gender_prediction) == 0:
				gender = "Woman"
			elif np.argmax(gender_prediction) == 1:
				gender = "Man"

			# resp_obj = json.loads(resp_obj)

			resp_obj = {
				'id': i,
				'age': np.round(apparent_age),
				'gender': gender,
				'emotion': emotion
			}

			resp_objects.append(resp_obj)

		return bbox_img, resp_objects



