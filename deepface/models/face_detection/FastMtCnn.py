# built-in dependencies
from typing import Any, Union, List

# 3rd party dependencies
import cv2
import numpy as np

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion


class FastMtCnnClient(Detector):
    """
    Fast MtCnn Detector from github.com/timesler/facenet-pytorch
    """

    def __init__(self):
        self.model = self.build_model()

    def detect_faces(self, img: Union[np.ndarray, List[np.ndarray]]) -> Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]:
        """
        Detect and align face with mtcnn

        Args:
            img (Union[np.ndarray, List[np.ndarray]]): pre-loaded image as numpy array or a list of those

        Returns:
            results (Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]): A list or a list of lists of FacialAreaRegion objects
        """
        if isinstance(img, np.ndarray):
            return self._process_single_image(img)
        elif isinstance(img, list):
            return [self._process_single_image(single_img) for single_img in img]
        else:
            raise ValueError("Input must be a numpy array or a list of numpy arrays.")

    def _process_single_image(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Helper function to detect faces in a single image.

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
        detections = self.model.detect(
            img_rgb, landmarks=True
        )  # returns boundingbox, prob, landmark
        if (
            detections is not None
            and len(detections) > 0
            and not any(detection is None for detection in detections)  # issue 1043
        ):
            for regions, confidence, eyes in zip(*detections):
                x, y, w, h = xyxy_to_xywh(regions)
                right_eye = eyes[0]
                left_eye = eyes[1]

                left_eye = tuple(int(i) for i in left_eye)
                right_eye = tuple(int(i) for i in right_eye)

                facial_area = FacialAreaRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    confidence=confidence,
                )
                resp.append(facial_area)

        return resp

    def build_model(self) -> Any:
        """
        Build a fast mtcnn face detector model
        Returns:
            model (Any)
        """
        # this is not a must dependency. do not import it in the global level.
        try:
            from facenet_pytorch import MTCNN as fast_mtcnn
            import torch
        except ModuleNotFoundError as e:
            raise ImportError(
                "FastMtcnn is an optional detector, ensure the library is installed. "
                "Please install using 'pip install facenet-pytorch'"
            ) from e

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        face_detector = fast_mtcnn(device=device)

        return face_detector


def xyxy_to_xywh(regions: Union[list, tuple]) -> tuple:
    """
    Convert (x1, y1, x2, y2) format to (x, y, w, h) format.
    Args:
        regions (list or tuple): facial area coordinates as x, y, x+w, y+h
    Returns:
        regions (tuple): facial area coordinates as x, y, w, h
    """
    x, y, x_plus_w, y_plus_h = regions[0], regions[1], regions[2], regions[3]
    w = x_plus_w - x
    h = y_plus_h - y
    return (int(x), int(y), int(w), int(h))
