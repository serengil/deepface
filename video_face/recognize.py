import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import cv2
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from deepface.commons import functions, realtime, distance as dst


def face_recognize(faces, db_embeddings, model, model_name, distance_metric):

    # threshold = dst.findThreshold(model_name, distance_metric) 
    embeddings = model.predict(faces)

    if distance_metric == 'cosine':
        distance = np.matmul(embeddings, np.transpose(db_embeddings))
    elif distance_metric == 'euclidean':
        distance = dst.findEuclideanDistance(embeddings, db_embeddings)
    elif distance_metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(embeddings), dst.l2_normalize(db_embeddings))

    shortest_distance = np.max(distance, axis=1)
    pred = np.argmax(distance, axis=1)
    return shortest_distance, pred



