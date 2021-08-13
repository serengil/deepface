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
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector


def face_det(frame, detector_backend, rec_model_input_size, smallest_faces=30):
    face_detector = FaceDetector.build_model(detector_backend)
    faces = FaceDetector.detect_faces(face_detector, detector_backend, frame, align = False)
    size_x = rec_model_input_size[0]; size_y = rec_model_input_size[1]
    detected_faces = []
    face_imgs = []
    for face, (x, y, w, h) in faces:
        if w > smallest_faces:
            # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1) #draw rectangle to main image
            detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = functions.preprocess_face(img = detected_face, target_size = (size_y, size_x), enforce_detection = False, detector_backend = 'opencv')
            detected_faces.append((x,y,w,h))
            face_imgs.append(detected_face)
    if len(face_imgs) > 0:
        face_imgs = np.vstack(face_imgs)
    return detected_faces, face_imgs