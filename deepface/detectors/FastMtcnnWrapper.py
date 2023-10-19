import cv2
from deepface.detectors import FaceDetector

# Link -> https://github.com/timesler/facenet-pytorch
# Examples https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch

def build_model():
    # Optional dependency
    from facenet_pytorch import MTCNN as fast_mtcnn


    face_detector = fast_mtcnn(image_size=160,
              thresholds=[0.6, 0.7, 0.7], # MTCNN thresholds
              post_process=True,
              device='cpu',
              select_largest=False, # return result in descending order
        )
    return face_detector

def xyxy_to_xywh(xyxy):
    """
    Convert xyxy format to xywh format.
    """
    x, y = xyxy[0], xyxy[1]
    w = xyxy[2] - x + 1
    h = xyxy[3] - y + 1
    return [x, y, w, h]

def detect_face(face_detector, img, align=True):

    resp = []

    detected_face = None
    img_region = [0, 0, img.shape[1], img.shape[0]]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
    detections = face_detector.detect(img_rgb, landmarks=True) # returns boundingbox, prob, landmark
    if len(detections[0]) > 0:

        for detection in zip(*detections):
            x, y, w, h = xyxy_to_xywh(detection[0])
            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
            img_region = [x, y, w, h]
            confidence = detection[1]

            if align:
                left_eye = detection[2][0]
                right_eye = detection[2][1]
                detected_face = FaceDetector.alignment_procedure(detected_face, left_eye, right_eye)

            resp.append((detected_face, img_region, confidence))

    return resp
