import os
from typing import Any, List
import cv2
import numpy as np
import gdown
from deepface.commons import functions
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.modules import detection
from deepface.commons.logger import Logger

logger = Logger(module="detectors.YunetWrapper")


class YuNetClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a yunet detector model
        Returns:
            model (Any)
        """

        opencv_version = cv2.__version__.split(".")

        if len(opencv_version) > 2 and int(opencv_version[0]) == 4 and int(opencv_version[1]) < 8:
            # min requirement: https://github.com/opencv/opencv_zoo/issues/172
            raise ValueError(f"YuNet requires opencv-python >= 4.8 but you have {cv2.__version__}")

        # pylint: disable=C0301
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        file_name = "face_detection_yunet_2023mar.onnx"
        home = functions.get_deepface_home()
        if os.path.isfile(home + f"/.deepface/weights/{file_name}") is False:
            logger.info(f"{file_name} will be downloaded...")
            output = home + f"/.deepface/weights/{file_name}"
            gdown.download(url, output, quiet=False)

        try:
            face_detector = cv2.FaceDetectorYN_create(
                home + f"/.deepface/weights/{file_name}", "", (0, 0)
            )
        except Exception as err:
            raise ValueError(
                "Exception while calling opencv.FaceDetectorYN_create module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err
        return face_detector

    def detect_faces(self, img: np.ndarray, align: bool = True) -> List[DetectedFace]:
        """
        Detect and align face with yunet
        Args:
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            results (List[DetectedFace]): A list of DetectedFace objects
                where each object contains:
            - img (np.ndarray): The detected face as a NumPy array.
            - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h
            - confidence (float): The confidence score associated with the detected face.
        """
        # FaceDetector.detect_faces does not support score_threshold parameter.
        # We can set it via environment variable.
        score_threshold = float(os.environ.get("yunet_score_threshold", "0.9"))
        resp = []
        detected_face = None
        faces = []
        height, width = img.shape[0], img.shape[1]
        # resize image if it is too large (Yunet fails to detect faces on large input sometimes)
        # I picked 640 as a threshold because it is the default value of max_size in Yunet.
        resized = False
        if height > 640 or width > 640:
            r = 640.0 / max(height, width)
            original_image = img.copy()
            img = cv2.resize(img, (int(width * r), int(height * r)))
            height, width = img.shape[0], img.shape[1]
            resized = True
        self.model.setInputSize((width, height))
        self.model.setScoreThreshold(score_threshold)
        _, faces = self.model.detect(img)
        if faces is None:
            return resp
        for face in faces:
            # pylint: disable=W0105
            """
            The detection output faces is a two-dimension array of type CV_32F,
            whose rows are the detected face instances, columns are the location
            of a face and 5 facial landmarks.
            The format of each row is as follows:
            x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
            x_rcm, y_rcm, x_lcm, y_lcm,
            where x1, y1, w, h are the top-left coordinates, width and height of
            the face bounding box,
            {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye,
            left eye, nose tip, the right corner and left corner of the mouth respectively.
            """
            (x, y, w, h, x_re, y_re, x_le, y_le) = list(map(int, face[:8]))

            # Yunet returns negative coordinates if it thinks part of
            # the detected face is outside the frame.
            # We set the coordinate to 0 if they are negative.
            x = max(x, 0)
            y = max(y, 0)
            if resized:
                img = original_image
                x, y, w, h = int(x / r), int(y / r), int(w / r), int(h / r)
                x_re, y_re, x_le, y_le = (
                    int(x_re / r),
                    int(y_re / r),
                    int(x_le / r),
                    int(y_le / r),
                )
            confidence = face[-1]
            confidence = f"{confidence:.2f}"
            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
            img_region = FacialAreaRegion(x=x, y=y, w=w, h=h)
            if align:
                detected_face = detection.align_face(detected_face, (x_re, y_re), (x_le, y_le))

            detected_face_obj = DetectedFace(
                img=detected_face, facial_area=img_region, confidence=confidence
            )
            resp.append(detected_face_obj)
        return resp
