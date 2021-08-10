from deepface import DeepFace

# DeepFace.stream("dataset") #opencv
#DeepFace.stream("dataset", detector_backend = 'opencv')
DeepFace.stream("D:/face/320_database/", detector_backend = 'ssd', enable_face_analysis=False)
# DeepFace.stream("D:/face/320_database/", detector_backend = 'mtcnn')
#DeepFace.stream("dataset", detector_backend = 'dlib')
# DeepFace.stream("D:/face/320_database/", detector_backend = 'retinaface')
