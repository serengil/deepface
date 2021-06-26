from deepface.detectors import OpenCvWrapper, SsdWrapper, DlibWrapper, MtcnnWrapper, RetinaFaceWrapper
from PIL import Image
import math
import numpy as np
from deepface.commons import distance

def build_model(detector_backend):

    global face_detector_obj #singleton design pattern

    backends = {
        'opencv': OpenCvWrapper.build_model,
        'ssd': SsdWrapper.build_model,
        'dlib': DlibWrapper.build_model,
        'mtcnn': MtcnnWrapper.build_model,
        'retinaface': RetinaFaceWrapper.build_model
    }

    if not "face_detector_obj" in globals():
        face_detector_obj = {}

    if not detector_backend in face_detector_obj.keys():
        face_detector = backends.get(detector_backend)

        if face_detector:
            face_detector = face_detector()
            face_detector_obj[detector_backend] = face_detector
            #print(detector_backend," built")
        else:
            raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face_detector_obj[detector_backend]

def detect_face(face_detector, detector_backend, img, align = True):

    backends = {
        'opencv': OpenCvWrapper.detect_face,
        'ssd': SsdWrapper.detect_face,
        'dlib': DlibWrapper.detect_face,
        'mtcnn': MtcnnWrapper.detect_face,
        'retinaface': RetinaFaceWrapper.detect_face
    }

    detect_face = backends.get(detector_backend)

    if detect_face:
        face, region = detect_face(face_detector, img, align)
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face, region

def alignment_procedure(img, left_eye, right_eye):

	#this function aligns given face in img based on left and right eye coordinates

	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye

	#-----------------------
	#find rotation direction

	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1 #rotate same direction to clock
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1 #rotate inverse direction of clock

	#-----------------------
	#find length of triangle edges

	a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

	#-----------------------

	#apply cosine rule

	if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a) #angle in radian
		angle = (angle * 180) / math.pi #radian to degree

		#-----------------------
		#rotate base image

		if direction == -1:
			angle = 90 - angle

		img = Image.fromarray(img)
		img = np.array(img.rotate(direction * angle))

	#-----------------------

	return img #return img anyway
