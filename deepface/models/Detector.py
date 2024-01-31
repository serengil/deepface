from typing import List
from abc import ABC, abstractmethod
import numpy as np

# Notice that all facial detector models must be inherited from this class


# pylint: disable=unnecessary-pass, too-few-public-methods
class Detector(ABC):
    @abstractmethod
    def detect_faces(
        self, img: np.ndarray, align: bool = True, expand_percentage: int = 0
    ) -> List["DetectedFace"]:
        """
        Interface for detect and align face

        Args:
            img (np.ndarray): pre-loaded image as numpy array

            align (bool): flag to enable or disable alignment after detection (default is True)

            expand_percentage (int): expand detected facial area with a percentage

        Returns:
            results (List[Tuple[DetectedFace]): A list of DetectedFace objects
                where each object contains:

            - img (np.ndarray): The detected face as a NumPy array.

            - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h

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
