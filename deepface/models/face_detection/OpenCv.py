# built-in dependencies
import os
from typing import Any, List, Union
import logging

# 3rd party dependencies
import cv2
import numpy as np

#project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion

logger = logging.getLogger(__name__)

class OpenCvClient(Detector):
    """
    Class to cover common face detection functionalitiy for OpenCv backend
    """

    def __init__(self):
        self.model = self.build_model()
        self.supports_batch_detection = self._supports_batch_detection()

    def build_model(self):
        """
        Build opencv's face and eye detector models
        Returns:
            model (dict): including face_detector and eye_detector keys
        """
        detector = {}
        detector["face_detector"] = self.__build_cascade("haarcascade")
        detector["eye_detector"] = self.__build_cascade("haarcascade_eye")
        return detector

    def _supports_batch_detection(self) -> bool:
        supports_batch_detection = os.getenv(
            "ENABLE_OPENCV_BATCH_DETECTION", "false"
        ).lower() == "true"
        if not supports_batch_detection:
            logger.warning(
                "Batch detection is disabled for opencv by default "
                "since the results are not consistent with single image detection. "
                "You can force enable it by setting the environment variable "
                "ENABLE_OPENCV_BATCH_DETECTION to true."
            )
        return supports_batch_detection

    def detect_faces(
        self,
        img: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]:
        """
        Detect and align face with opencv

        Args:
            img (Union[np.ndarray, List[np.ndarray]]):
            Pre-loaded image as numpy array or a list of those

        Returns:
            results (Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]): 
            A list or a list of lists of FacialAreaRegion objects
        """
        if isinstance(img, np.ndarray):
            imgs = [img]
        elif self.supports_batch_detection:
            imgs = img
        else:
            return [self.detect_faces(single_img) for single_img in img]

        batch_results = []

        for single_img in imgs:
            resp = []
            detected_face = None
            faces = []
            try:
                faces, _, scores = self.model["face_detector"].detectMultiScale3(
                    single_img, 1.1, 10, outputRejectLevels=True
                )
            except:
                pass

            if len(faces) > 0:
                for (x, y, w, h), confidence in zip(faces, scores):
                    detected_face = single_img[int(y):int(y + h), int(x):int(x + w)]
                    left_eye, right_eye = self.find_eyes(img=detected_face)

                    if left_eye is not None:
                        left_eye = (int(x + left_eye[0]), int(y + left_eye[1]))
                    if right_eye is not None:
                        right_eye = (int(x + right_eye[0]), int(y + right_eye[1]))

                    facial_area = FacialAreaRegion(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        left_eye=left_eye,
                        right_eye=right_eye,
                        confidence=(100 - confidence) / 100,
                    )
                    resp.append(facial_area)

            batch_results.append(resp)

        return batch_results if len(batch_results) > 1 else batch_results[0]

    def find_eyes(self, img: np.ndarray) -> tuple:
        """
        Find the left and right eye coordinates of given image
        Args:
            img (np.ndarray): given image
        Returns:
            left and right eye (tuple)
        """
        left_eye = None
        right_eye = None

        # if image has unexpectedly 0 dimension then skip alignment
        if img.shape[0] == 0 or img.shape[1] == 0:
            return left_eye, right_eye

        detected_face_gray = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY
        )  # eye detector expects gray scale image

        eyes = self.model["eye_detector"].detectMultiScale(detected_face_gray, 1.1, 10)

        # ----------------------------------------------------------------

        # opencv eye detection module is not strong. it might find more than 2 eyes!
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
                right_eye = eye_1
                left_eye = eye_2
            else:
                right_eye = eye_2
                left_eye = eye_1

            # -----------------------
            # find center of eyes
            left_eye = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2)),
            )
            right_eye = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2)),
            )
        return left_eye, right_eye

    def __build_cascade(self, model_name="haarcascade") -> Any:
        """
        Build a opencv face&eye detector models
        Returns:
            model (Any)
        """
        opencv_path = self.__get_opencv_path()
        if model_name == "haarcascade":
            face_detector_path = os.path.join(opencv_path, "haarcascade_frontalface_default.xml")
            if not os.path.isfile(face_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    face_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(face_detector_path)

        elif model_name == "haarcascade_eye":
            eye_detector_path = os.path.join(opencv_path, "haarcascade_eye.xml")
            if not os.path.isfile(eye_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    eye_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(eye_detector_path)

        else:
            raise ValueError(f"unimplemented model_name for build_cascade - {model_name}")

        return detector

    def __get_opencv_path(self) -> str:
        """
        Returns where opencv installed
        Returns:
            installation_path (str)
        """
        return os.path.join(os.path.dirname(cv2.__file__), "data")
