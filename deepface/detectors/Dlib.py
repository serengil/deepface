from typing import List
import os
import bz2
import gdown
import numpy as np
from deepface.commons import functions
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger(module="detectors.DlibWrapper")


class DlibClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> dict:
        """
        Build a dlib hog face detector model
        Returns:
            model (Any)
        """
        home = functions.get_deepface_home()

        # this is not a must dependency. do not import it in the global level.
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional detector, ensure the library is installed."
                "Please install using 'pip install dlib' "
            ) from e

        # check required file exists in the home/.deepface/weights folder
        if os.path.isfile(home + "/.deepface/weights/shape_predictor_5_face_landmarks.dat") != True:

            file_name = "shape_predictor_5_face_landmarks.dat.bz2"
            logger.info(f"{file_name} is going to be downloaded")

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

    def detect_faces(self, img: np.ndarray, align: bool = True) -> List[DetectedFace]:
        """
        Detect and align face with dlib
        Args:
            face_detector (Any): dlib face detector object
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            results (List[DetectedFace]): A list of DetectedFace objects
                where each object contains:
            - img (np.ndarray): The detected face as a NumPy array.
            - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h
            - confidence (float): The confidence score associated with the detected face.
        """
        # this is not a must dependency. do not import it in the global level.
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional detector, ensure the library is installed."
                "Please install using 'pip install dlib' "
            ) from e

        resp = []

        sp = self.model["sp"]

        detected_face = None

        face_detector = self.model["face_detector"]

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

                img_region = FacialAreaRegion(x=left, y=right, w=right - left, h=bottom - top)
                confidence = scores[idx]

                if align:
                    img_shape = sp(img, detections[idx])
                    detected_face = dlib.get_face_chip(img, img_shape, size=detected_face.shape[0])

                detected_face_obj = DetectedFace(
                    img=detected_face, facial_area=img_region, confidence=confidence
                )

                resp.append(detected_face_obj)

        return resp
