import cv2
from deepface.detectors import FaceDetector

def build_model():
	from mtcnn import MTCNN
	face_detector = MTCNN()
	return face_detector

def detect_face(face_detector, img, align = True):

	detected_face = None
	img_region = [0, 0, img.shape[0], img.shape[1]]

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #mtcnn expects RGB but OpenCV read BGR
	detections = face_detector.detect_faces(img_rgb)

	if len(detections) > 0:
		detection = detections[0]
		x, y, w, h = detection["box"]
		detected_face = img[int(y):int(y+h), int(x):int(x+w)]
		img_region = [x, y, w, h]

		keypoints = detection["keypoints"]
		left_eye = keypoints["left_eye"]
		right_eye = keypoints["right_eye"]

		if align:
			detected_face = FaceDetector.alignment_procedure(detected_face, left_eye, right_eye)

	return detected_face, img_region
