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


def analysis(db_path, model_name = 'VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', source = 0, smallest_faces=30):

    face_detector = FaceDetector.build_model(detector_backend)
    print("Detector backend is ", detector_backend)
    model = DeepFace.build_model(model_name)
    print(model_name, "is built")    
    threshold = dst.findThreshold(model_name, distance_metric)
    input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

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
            
    df = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
    df['distance_metric'] = distance_metric
    frame_count = 0
    start_time = time.time()

    cap = cv2.VideoCapture(source)
    while True:
        frame_count += 1
        ret, img = cap.read()
        
        if img is None:
            break

        raw_img = img.copy()
        faces = []

        if frame_count % 10 == 0:
            faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)
        detected_faces = []

        
        for face, (x, y, w, h) in faces:
            if w > smallest_faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1) #draw rectangle to main image

                detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

                detected_faces.append((x,y,w,h))
            
        base_img = raw_img.copy()
        for detected_face in detected_faces:
            x = detected_face[0]; y = detected_face[1]
            w = detected_face[2]; h = detected_face[3]
            custom_face = base_img[y:y+h, x:x+w]
            custom_face = functions.preprocess_face(img = custom_face, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = 'opencv')
            if custom_face.shape[1:3] == input_shape:
                if df.shape[0] > 0: #if there are images to verify, apply face recognition
                    img1_representation = model.predict(custom_face)[0,:]

                    #print(freezed_frame," - ",img1_representation[0:5])

                    def findDistance(row):
                        distance_metric = row['distance_metric']
                        img2_representation = row['embedding']

                        distance = 1000 #initialize very large value
                        if distance_metric == 'cosine':
                            distance = dst.findCosineDistance(img1_representation, img2_representation)
                        elif distance_metric == 'euclidean':
                            distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                        elif distance_metric == 'euclidean_l2':
                            distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))

                        return distance

                    df['distance'] = df.apply(findDistance, axis = 1)
                    df = df.sort_values(by = ["distance"])

                    candidate = df.iloc[0]
                    employee_name = candidate['employee']
                    best_distance = candidate['distance']

                    # if best_distance <= threshold: 
                    if best_distance <= 0.25: 
                        label = employee_name.split("/")[-1].replace(".jpg", "")
                        # label = re.sub('[0-9]', '', label)
                        cv2.putText(img, label, (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    else:
                        label = 'unknown'
                        cv2.putText(img, label, (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    print("best_distance:{}, threshhold:{}, label:{}".format(best_distance, threshold, label))
                        
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break
    cap.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':
    analysis("D:/face/320_no_mask/", model_name = 'VGG-Face', detector_backend = 'ssd', 
            # source='rtsp://admin:123456@192.168.123.235:554/stream1', 
            source="C:/Users/DELL/Desktop/face_det/yg.mp4", 
            distance_metric='cosine', smallest_faces=20)
