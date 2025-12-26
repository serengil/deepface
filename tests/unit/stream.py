from deepface import DeepFace

DeepFace.stream("dataset", enable_face_analysis=False, anti_spoofing=True)  # opencv
# DeepFace.stream("dataset", detector_backend = 'opencv')
# DeepFace.stream("dataset", detector_backend = 'ssd')
# DeepFace.stream("dataset", detector_backend = 'mtcnn')
# DeepFace.stream("dataset", detector_backend = 'dlib')
# DeepFace.stream("dataset", detector_backend = 'retinaface')
