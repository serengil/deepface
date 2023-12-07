import os
import gdown
import cv2
import pandas as pd
from deepface.detectors import OpenCvWrapper
from deepface.commons import functions
from deepface.commons.logger import Logger

logger = Logger(module="detectors.SsdWrapper")

# pylint: disable=line-too-long


def build_model():

    home = functions.get_deepface_home()

    # model structure
    if os.path.isfile(home + "/.deepface/weights/deploy.prototxt") != True:

        logger.info("deploy.prototxt will be downloaded...")

        url = "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt"

        output = home + "/.deepface/weights/deploy.prototxt"

        gdown.download(url, output, quiet=False)

    # pre-trained weights
    if os.path.isfile(home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel") != True:

        logger.info("res10_300x300_ssd_iter_140000.caffemodel will be downloaded...")

        url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

        output = home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"

        gdown.download(url, output, quiet=False)

    face_detector = cv2.dnn.readNetFromCaffe(
        home + "/.deepface/weights/deploy.prototxt",
        home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel",
    )

    eye_detector = OpenCvWrapper.build_cascade("haarcascade_eye")

    detector = {}
    detector["face_detector"] = face_detector
    detector["eye_detector"] = eye_detector

    return detector


def detect_face(detector, img, align=True):

    resp = []

    detected_face = None
    img_region = [0, 0, img.shape[1], img.shape[0]]

    ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

    target_size = (300, 300)

    base_img = img.copy()  # we will restore base_img to img later

    original_size = img.shape

    img = cv2.resize(img, target_size)

    aspect_ratio_x = original_size[1] / target_size[1]
    aspect_ratio_y = original_size[0] / target_size[0]

    imageBlob = cv2.dnn.blobFromImage(image=img)

    face_detector = detector["face_detector"]
    face_detector.setInput(imageBlob)
    detections = face_detector.forward()

    detections_df = pd.DataFrame(detections[0][0], columns=ssd_labels)

    detections_df = detections_df[detections_df["is_face"] == 1]  # 0: background, 1: face
    detections_df = detections_df[detections_df["confidence"] >= 0.90]

    detections_df["left"] = (detections_df["left"] * 300).astype(int)
    detections_df["bottom"] = (detections_df["bottom"] * 300).astype(int)
    detections_df["right"] = (detections_df["right"] * 300).astype(int)
    detections_df["top"] = (detections_df["top"] * 300).astype(int)

    if detections_df.shape[0] > 0:

        for _, instance in detections_df.iterrows():

            left = instance["left"]
            right = instance["right"]
            bottom = instance["bottom"]
            top = instance["top"]

            detected_face = base_img[
                int(top * aspect_ratio_y) : int(bottom * aspect_ratio_y),
                int(left * aspect_ratio_x) : int(right * aspect_ratio_x),
            ]
            img_region = [
                int(left * aspect_ratio_x),
                int(top * aspect_ratio_y),
                int(right * aspect_ratio_x) - int(left * aspect_ratio_x),
                int(bottom * aspect_ratio_y) - int(top * aspect_ratio_y),
            ]
            confidence = instance["confidence"]

            if align:
                detected_face = OpenCvWrapper.align_face(detector["eye_detector"], detected_face)

            resp.append((detected_face, img_region, confidence))

    return resp
