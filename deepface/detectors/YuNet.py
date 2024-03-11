import os
from typing import Any, List
import cv2
import numpy as np
import gdown
from deepface.commons import folder_utils
from deepface.models.Detector import Detector, FacialAreaRegion
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
        if not len(opencv_version) >= 2:
            raise ValueError(
                f"OpenCv's version must have major and minor values but it is {opencv_version}"
            )

        opencv_version_major = int(opencv_version[0])
        opencv_version_minor = int(opencv_version[1])

        if opencv_version_major < 4 or (opencv_version_major == 4 and opencv_version_minor < 8):
            # min requirement: https://github.com/opencv/opencv_zoo/issues/172
            raise ValueError(f"YuNet requires opencv-python >= 4.8 but you have {cv2.__version__}")

        # pylint: disable=C0301
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        file_name = "face_detection_yunet_2023mar.onnx"
        home = folder_utils.get_deepface_home()
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

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with yunet

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        # FaceDetector.detect_faces does not support score_threshold parameter.
        # We can set it via environment variable.
        score_threshold = float(os.environ.get("yunet_score_threshold", "0.9"))
        resp = []
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
            left_eye = (x_re, y_re)
            right_eye = (x_le, y_le)

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
            confidence = float(face[-1])

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=confidence,
                left_eye=left_eye,
                right_eye=right_eye,
            )
            resp.append(facial_area)
        return resp
