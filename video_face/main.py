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
from database_encode import encode
from detect import face_det
from recognize import face_recognize


def main(db_path, model_name = 'VGG-Face', detector_backend = 'retinaface', distance_metric = 'cosine', source = 0, smallest_faces=30):
    model = DeepFace.build_model(model_name)
    print(model_name, "is built")
    input_shape = functions.find_input_shape(model)
    threshold = dst.findThreshold(model_name, distance_metric)
    employees, db_embeddings = encode(db_path, model_name=model_name, detector_backend=detector_backend)
    db_embeddings = [db_embeddings[i][1] for i in range(len(db_embeddings))]
    db_embeddings = np.vstack(db_embeddings)
    frame_count = 0
    cap = cv2.VideoCapture(source)
    while True:
        frame_count += 1
        ret, frame = cap.read()

        if frame is None:
            break

        if frame_count % 10 == 0:
            detected_faces, face_imgs = face_det(frame, detector_backend, rec_model_input_size=input_shape, smallest_faces=30)
            if len(face_imgs) > 0:
                shortest_distance, pred = face_recognize(face_imgs, db_embeddings, model, model_name, distance_metric)
                labels = [employees[i] for i in pred]
                for i, detected_face in enumerate(detected_faces):
                    x = detected_face[0]; y = detected_face[1]
                    w = detected_face[2]; h = detected_face[3]
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
                    if shortest_distance[i] <= threshold: 
                        label = labels[i]
                        cv2.putText(frame, label, (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        label = 'unknown'
                        cv2.putText(frame, label, (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    print("best_distance:{}, threshhold:{}, label:{}".format(shortest_distance, threshold, label))
        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main("D:/face/320_no_mask/", model_name = 'VGG-Face', detector_backend = 'ssd', 
            # source='rtsp://admin:123456@192.168.123.235:554/stream1', 
            source="C:/Users/DELL/Desktop/face_det/320.mp4", 
            distance_metric='cosine', smallest_faces=20)