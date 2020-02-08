import matplotlib.pyplot as plt
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
import time
import os

from deepface.basemodels import VGGFace, OpenFace, Facenet
from deepface.commons import functions, distance as dst

def verify(img1_path, img2_path
	, model_name ='VGG-Face', distance_metric = 'cosine'):
	
	tic = time.time()
	
	if os.path.isfile(img1_path) != True:
		raise ValueError("Confirm that ",img1_path," exists")
	
	if os.path.isfile(img2_path) != True:
		raise ValueError("Confirm that ",img2_path," exists")
		
	#-------------------------
	
	#print("Face verification will be applied on ",model_name," model and ",distance_metric," metric")
	
	functions.validateInputs(model_name, distance_metric)
	
	#-------------------------
	
	#tuned thresholds for model and metric pair
	threshold = functions.findThreshold(model_name, distance_metric)
	
	#-------------------------
	
	if model_name == 'VGG-Face':
		model = VGGFace.loadModel()
		input_shape = (224, 224)	
	
	elif model_name == 'OpenFace':
		model = OpenFace.loadModel()
		input_shape = (96, 96)
	
	elif model_name == 'Facenet':
		model = Facenet.loadModel()
		input_shape = (160, 160)
	
	#-------------------------
	#crop face
	
	img1 = functions.detectFace(img1_path, input_shape)
	img2 = functions.detectFace(img2_path, input_shape)
	
	#-------------------------
	#TO-DO: Apply face alignment here. Experiments show that aligment increases accuracy 1%.
	
	#-------------------------
	#find embeddings
	
	img1_representation = model.predict(img1)[0,:]
	img2_representation = model.predict(img2)[0,:]
	
	#-------------------------
	#find distances between embeddings
	
	if distance_metric == 'cosine':
		distance = dst.findCosineDistance(img1_representation, img2_representation)
	elif distance_metric == 'euclidean':
		distance = dst.findEuclideanDistance(img1_representation, img2_representation)
	elif distance_metric == 'euclidean_l2':
		distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
	
	#-------------------------
	#decision
	
	if distance <= threshold:
		identified =  True
	else:
		identified =  False
	
	#-------------------------
	
	plot = False
	
	if plot:
		label = "Distance is "+str(round(distance, 2))
		
		fig = plt.figure()
		fig.add_subplot(1,2, 1)
		plt.imshow(img1[0][:, :, ::-1])
		plt.xticks([]); plt.yticks([])
		fig.add_subplot(1,2, 2)
		plt.imshow(img2[0][:, :, ::-1])
		plt.xticks([]); plt.yticks([])
		fig.suptitle(label, fontsize=17)
		plt.show(block=True)
	
	#-------------------------
	
	toc = time.time()
	
	#print("identification lasts ",toc-tic," seconds")
	
	#Return a tuple. First item is the identification result based on tuned threshold.
	#Second item is the threshold. You might want to customize this threshold to identify faces.
	return (identified, distance, threshold)

#---------------------------

functions.initializeFolder()

#---------------------------