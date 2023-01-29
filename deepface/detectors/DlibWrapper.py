import os
import bz2
import gdown
from deepface.commons import functions


def build_model():

    home = functions.get_deepface_home()

    import dlib  # this requirement is not a must that's why imported here

    # check required file exists in the home/.deepface/weights folder
    if os.path.isfile(home + "/.deepface/weights/shape_predictor_5_face_landmarks.dat") != True:

        file_name = "shape_predictor_5_face_landmarks.dat.bz2"
        print(f"{file_name} is going to be downloaded")

        url = f"http://dlib.net/files/{file_name}"
        output = f"{home}/.deepface/weights/{file_name}"

        gdown.download(url, output, quiet=False)

        zipfile = bz2.BZ2File(output)
        data = zipfile.read()
        newfilepath = output[:-4]  # discard .bz2 extension
        with open(newfilepath, "wb") as f:
            f.write(data)

    face_detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(home + "/.deepface/weights/shape_predictor_5_face_landmarks.dat")

    detector = {}
    detector["face_detector"] = face_detector
    detector["sp"] = sp
    return detector


def detect_face(detector, img, align=True):

    import dlib  # this requirement is not a must that's why imported here

    resp = []

    sp = detector["sp"]

    detected_face = None

    img_region = [0, 0, img.shape[1], img.shape[0]]

    face_detector = detector["face_detector"]

    # note that, by design, dlib's fhog face detector scores are >0 but not capped at 1
    detections, scores, _ = face_detector.run(img, 1)

    if len(detections) > 0:

        for idx, d in enumerate(detections):
            left = d.left()
            right = d.right()
            top = d.top()
            bottom = d.bottom()

            # detected_face = img[top:bottom, left:right]
            detected_face = img[
                max(0, top) : min(bottom, img.shape[0]), max(0, left) : min(right, img.shape[1])
            ]

            img_region = [left, top, right - left, bottom - top]
            confidence = scores[idx]

            if align:
                img_shape = sp(img, detections[idx])
                detected_face = dlib.get_face_chip(img, img_shape, size=detected_face.shape[0])

            resp.append((detected_face, img_region, confidence))

    return resp
