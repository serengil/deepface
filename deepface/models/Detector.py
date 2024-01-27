from typing import List
from abc import ABC, abstractmethod
import numpy as np

# Notice that all facial detector models must be inherited from this class


# pylint: disable=unnecessary-pass, too-few-public-methods
class Detector(ABC):
    @abstractmethod
    def detect_faces(self, img: np.ndarray, align: bool = True) -> List["DetectedFace"]:
        """
        Detect faces from a given image
        Args:
            img (np.ndarray): pre-loaded image as a NumPy array
            align (bool): enable or disable alignment after face detection
        Returns:
            results (List[DetectedFace]): A list of DetectedFace object
                where each object contains:
                - face (np.ndarray): The detected face as a NumPy array.
                - face_region (List[float]): The image region represented as
                    a list of floats e.g. [x, y, w, h]
                - confidence (float): The confidence score associated with the detected face.
        """
        pass


class FacialAreaRegion:
    x: int
    y: int
    w: int
    h: int

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class DetectedFace:
    img: np.ndarray
    facial_area: FacialAreaRegion
    confidence: float

    def __init__(self, img: np.ndarray, facial_area: FacialAreaRegion, confidence: float):
        self.img = img
        self.facial_area = facial_area
        self.confidence = confidence
