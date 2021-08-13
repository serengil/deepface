import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import cv2
import time
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace
from deepface.commons import functions, distance as dst


def encode(db_path, model_name='VGG-Face', detector_backend='retinaface'):
    model = DeepFace.build_model(model_name)
    print(model_name, "is built")    
    # threshold = dst.findThreshold(model_name, distance_metric)
    input_shape = functions.find_input_shape(model); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

    text_color = (255, 255, 255)

    employees = []
    if os.path.isdir(db_path) == True:
        for r, d, f in os.walk(db_path):
            for file in f:
                if('.jpg' in file):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

    if os.path.isdir(db_path) == True:
        file_name = "representations_%s.pkl" % (model_name)
        file_name = file_name.replace("-", "_").lower()

        if os.path.exists(db_path+"/"+file_name):

            print("Representations for images in ",db_path," folder were previously stored in ", file_name, ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")

            f = open(db_path+'/'+file_name, 'rb')
            embeddings = pickle.load(f)

            print("There are ", len(embeddings)," representations found in ",file_name)

        else:
            if len(employees) == 0:
                print("WARNING: There is no image in this path ( ", db_path,") . Face recognition will not be performed.")

            if len(employees) > 0:                
                input_shape = functions.find_input_shape(model)
                input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

            tic = time.time()
            pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')
            embeddings = []
            for index in pbar:
                employee = employees[index]
                pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
                embedding = []
                img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = detector_backend)
                img_representation = model.predict(img)[0,:]
                embedding.append(employee)
                embedding.append(img_representation)
                embeddings.append(embedding)
            f = open(db_path+'/'+file_name, "wb")
            pickle.dump(embeddings, f)
            f.close()
            toc = time.time()
            print("Embeddings found for given data set in ", toc-tic," seconds")

    return employees, embeddings