from typing import List
import os
import bz2
import gdown
import numpy as np
from deepface.commons import folder_utils
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()


class DlibClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> dict:
        """
        Build a dlib hog face detector model
        Returns:
            model (Any)
        """
        home = folder_utils.get_deepface_home()

        # this is not a must dependency. do not import it in the global level.
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional detector, ensure the library is installed. "
                "Please install using 'pip install dlib'"
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

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with dlib

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        face_detector = self.model["face_detector"]

        # note that, by design, dlib's fhog face detector scores are >0 but not capped at 1
        detections, scores, _ = face_detector.run(img, 1)

        if len(detections) > 0:

            for idx, detection in enumerate(detections):
                left = detection.left()
                right = detection.right()
                top = detection.top()
                bottom = detection.bottom()

                y = int(max(0, top))
                h = int(min(bottom, img.shape[0]) - y)
                x = int(max(0, left))
                w = int(min(right, img.shape[1]) - x)

                shape = self.model["sp"](img, detection)

                right_eye = (
                    int((shape.part(2).x + shape.part(3).x) // 2),
                    int((shape.part(2).y + shape.part(3).y) // 2),
                )
                left_eye = (
                    int((shape.part(0).x + shape.part(1).x) // 2),
                    int((shape.part(0).y + shape.part(1).y) // 2),
                )

                # never saw confidence higher than +3.5 github.com/davisking/dlib/issues/761
                confidence = scores[idx]

                facial_area = FacialAreaRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    confidence=min(max(0, confidence), 1.0),
                )
                resp.append(facial_area)

        return resp
