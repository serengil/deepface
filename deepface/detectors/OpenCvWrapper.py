import os
import cv2
from deepface.detectors import FaceDetector


def build_model():
    detector = {}
    detector["face_detector"] = build_cascade("haarcascade")
    detector["eye_detector"] = build_cascade("haarcascade_eye")
    return detector


def build_cascade(model_name="haarcascade"):
    opencv_path = get_opencv_path()
    if model_name == "haarcascade":
        face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
        if os.path.isfile(face_detector_path) != True:
            raise ValueError(
                "Confirm that opencv is installed on your environment! Expected path ",
                face_detector_path,
                " violated.",
            )
        detector = cv2.CascadeClassifier(face_detector_path)

    elif model_name == "haarcascade_eye":
        eye_detector_path = opencv_path + "haarcascade_eye.xml"
        if os.path.isfile(eye_detector_path) != True:
            raise ValueError(
                "Confirm that opencv is installed on your environment! Expected path ",
                eye_detector_path,
                " violated.",
            )
        detector = cv2.CascadeClassifier(eye_detector_path)

    else:
        raise ValueError(f"unimplemented model_name for build_cascade - {model_name}")

    return detector


def detect_face(detector, img, align=True):
    resp = []

    detected_face = None
    img_region = [0, 0, img.shape[1], img.shape[0]]

    # Initialize faces and scores to empty lists
    faces = []
    scores = []
    try:
        # faces = detector["face_detector"].detectMultiScale(img, 1.3, 5)

        # note that, by design, opencv's haarcascade scores are >0 but not capped at 1
        faces, _, scores = detector["face_detector"].detectMultiScale3(
            img, 1.1, 10, outputRejectLevels=True
        )

    # except alone is too broad and will catch keyboard interrupts
    # Exception should be changed to something more specific in the future
    except Exception:  # pylint: disable=broad-except
        import traceback

        print(traceback.format_exc())
    
    # For each face and associated score, append face,
    # bounding box, and score to resp
    for (x, y, w, h), confidence in zip(faces, scores):
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

        if align:
            detected_face = align_face(detector["eye_detector"], detected_face)

        img_region = [x, y, w, h]

        resp.append((detected_face, img_region, confidence))

    return resp


def align_face(eye_detector, img):
    detected_face_gray = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )  # eye detector expects gray scale image

    # eyes = eye_detector.detectMultiScale(detected_face_gray, 1.3, 5)
    eyes = eye_detector.detectMultiScale(detected_face_gray, 1.1, 10)

    # ----------------------------------------------------------------

    # opencv eye detectin module is not strong. it might find more than 2 eyes!
    # besides, it returns eyes with different order in each call (issue 435)
    # this is an important issue because opencv is the default detector and ssd also uses this
    # find the largest 2 eye. Thanks to @thelostpeace

    eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

    # ----------------------------------------------------------------

    if len(eyes) >= 2:
        # decide left and right eye

        eye_1 = eyes[0]
        eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        # -----------------------
        # find center of eyes
        left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        right_eye = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
        img = FaceDetector.alignment_procedure(img, left_eye, right_eye)
    return img  # return img anyway


def get_opencv_path():
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    return path + "/data/"
