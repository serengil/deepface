from deepface.basemodels import VGGFace
import os
from pathlib import Path
import gdown
import numpy as np

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	from keras.models import Model, Sequential
	from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Convolution2D, Flatten, Activation

def loadModel(url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'):

	model = VGGFace.baseModel()

	#--------------------------

	classes = 2
	base_model_output = Sequential()
	base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	#--------------------------

	gender_model = Model(inputs=model.input, outputs=base_model_output)

	#--------------------------

	#load weights

	home = str(Path.home())

	if os.path.isfile(home+'/.deepface/weights/gender_model_weights.h5') != True:
		print("gender_model_weights.h5 will be downloaded...")

		output = home+'/.deepface/weights/gender_model_weights.h5'
		gdown.download(url, output, quiet=False)

	gender_model.load_weights(home+'/.deepface/weights/gender_model_weights.h5')

	return gender_model

	#--------------------------
