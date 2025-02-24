from typing import List, Tuple, Optional, Union
from abc import ABC
from dataclasses import dataclass
import numpy as np


# pylint: disable=unnecessary-pass, too-few-public-methods, too-many-instance-attributes
@dataclass
class FacialAreaRegion:
    """
    Initialize a Face object.

    Args:
        x (int): The x-coordinate of the top-left corner of the bounding box.
        y (int): The y-coordinate of the top-left corner of the bounding box.
        w (int): The width of the bounding box.
        h (int): The height of the bounding box.
        left_eye (tuple): The coordinates (x, y) of the left eye with respect to
            the person instead of observer. Default is None.
        right_eye (tuple): The coordinates (x, y) of the right eye with respect to
            the person instead of observer. Default is None.
        confidence (float, optional): Confidence score associated with the face detection.
            Default is None.
    """

    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    nose: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None


@dataclass
class DetectedFace:
    """
    Initialize detected face object.

    Args:
        img (np.ndarray): detected face image as numpy array
        facial_area (FacialAreaRegion): detected face's metadata (e.g. bounding box)
        confidence (float): confidence score for face detection
    """

    img: np.ndarray
    facial_area: FacialAreaRegion
    confidence: float


# Notice that all facial detector models must be inherited from this class

class Detector(ABC):

    def detect_faces(
        self,
        img: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]:
        """
        Detect and align faces in an image or a list of images

        Args:
            img (Union[np.ndarray, List[np.ndarray]]): 
            pre-loaded image as numpy array or a list of those

        Returns:
            results (Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]): 
            A list or a list of lists of FacialAreaRegion objects
        """
        is_batched_input = isinstance(img, list)
        if not is_batched_input:
            img = [img]
        results = [self._process_single_image(single_img) for single_img in img]
        if not is_batched_input:
            return results[0]
        return results

    def _process_single_image(
        self,
        img: np.ndarray
    ) -> List[FacialAreaRegion]:
        """
        Interface for detect and align faces in a single image

        Args:
            img (Union[np.ndarray, List[np.ndarray]]): 
            Pre-loaded image as numpy array or a list of those

        Returns:
            results (Union[List[List[FacialAreaRegion]], List[FacialAreaRegion]]): 
            A list or a list of lists of FacialAreaRegion objects
                where each object contains:

            - facial_area (FacialAreaRegion): The facial area region represented
                as x, y, w, h, left_eye and right_eye. left eye and right eye are
                eyes on the left and right respectively with respect to the person
                instead of observer.
        """
        raise NotImplementedError(
            "Subclasses that do not implement batch detection must implement this method"
        )
