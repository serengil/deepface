from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np

# Notice that all facial detector models must be inherited from this class


# pylint: disable=unnecessary-pass, too-few-public-methods
class Detector(ABC):
    @abstractmethod
    def detect_faces(self, img: np.ndarray) -> List["FacialAreaRegion"]:
        """
        Interface for detect and align face

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
                where each object contains:

            - facial_area (FacialAreaRegion): The facial area region represented
                as x, y, w, h, left_eye and right_eye
        """
        pass


class FacialAreaRegion:
    x: int
    y: int
    w: int
    h: int
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    confidence: float

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        left_eye: Optional[Tuple[int, int]] = None,
        right_eye: Optional[Tuple[int, int]] = None,
        confidence: Optional[float] = None,
    ):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.confidence = confidence


class DetectedFace:
    img: np.ndarray
    facial_area: FacialAreaRegion
    confidence: float

    def __init__(self, img: np.ndarray, facial_area: FacialAreaRegion, confidence: float):
        self.img = img
        self.facial_area = facial_area
        self.confidence = confidence
