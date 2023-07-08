import cv2
import os
import gdown
from deepface.detectors import FaceDetector
from deepface.commons import functions


def build_model():
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    file_name = "face_detection_yunet_2023mar.onnx"
    home = functions.get_deepface_home()
    if os.path.isfile(home + f"/.deepface/weights/{file_name}") is False:
        print(f"{file_name} will be downloaded...")
        output = home + f"/.deepface/weights/{file_name}"
        gdown.download(url, output, quiet=False)
    face_detector = cv2.FaceDetectorYN_create(
        home + f"/.deepface/weights/{file_name}", "", (0, 0)
    )
    return face_detector


def detect_face(detector, image, align=True, score_threshold=0.9):
    # FaceDetector.detect_faces does not support score_threshold parameter.
    # We can set it via environment variable.
    score_threshold = os.environ.get("yunet_score_threshold", score_threshold)
    resp = []
    detected_face = None
    img_region = [0, 0, image.shape[1], image.shape[0]]
    faces = []
    height, width = image.shape[0], image.shape[1]
    # resize image if it is too large (Yunet fails to detect faces on large input sometimes)
    # I picked 640 as a threshold because it is the default value of max_size in Yunet.
    if height > 640 or width > 640:
        r = 640.0 / max(height, width)
        image = cv2.resize(image, (int(width * r), int(height * r)))
        height, width = image.shape[0], image.shape[1]
    detector.setInputSize((width, height))
    detector.setScoreThreshold(score_threshold)
    _, faces = detector.detect(image)
    if faces is None:
        return resp
    for face in faces:
        """
        The detection output faces is a two-dimension array of type CV_32F,
        whose rows are the detected face instances, columns are the location of a face and 5 facial landmarks.
        The format of each row is as follows:
        x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm,
        where x1, y1, w, h are the top-left coordinates, width and height of the face bounding box,
        {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye, left eye, nose tip, the right corner and left corner of the mouth respectively.
        """
        (x, y, w, h, x_re, y_re, x_le, y_le) = list(map(int, face[:8]))
        confidence = face[-1]
        confidence = "{:.2f}".format(confidence)
        detected_face = image[int(y) : int(y + h), int(x) : int(x + w)]
        img_region = [x, y, w, h]
        if align:
            detected_face = yunet_align_face(detected_face, x_re, y_re, x_le, y_le)
        resp.append((detected_face, img_region, confidence))
    return resp


# x_re, y_re, x_le, y_le stands for the coordinates of right eye, left eye
def yunet_align_face(img, x_re, y_re, x_le, y_le):
    img = FaceDetector.alignment_procedure(img, (x_le, y_le), (x_re, y_re))
    return img
