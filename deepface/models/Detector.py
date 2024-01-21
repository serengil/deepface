from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np

# Notice that all facial detector models must be inherited from this class


# pylint: disable=unnecessary-pass, too-few-public-methods
class Detector(ABC):
    @abstractmethod
    def detect_faces(
        self, img: np.ndarray, align: bool = True
    ) -> List[Tuple[np.ndarray, List[float], float]]:
        """
        Detect faces from a given image
        Args:
            img (np.ndarray): pre-loaded image as a NumPy array
            align (bool): enable or disable alignment after face detection
        Returns:
            results (List[Tuple[np.ndarray, List[float], float]]): A list of tuples
                where each tuple contains:
                - detected_face (np.ndarray): The detected face as a NumPy array.
                - face_region (List[float]): The image region represented as
                    a list of floats e.g. [x, y, w, h]
                - confidence (float): The confidence score associated with the detected face.

        Example:
            results = [
                (array(..., dtype=uint8), [110, 60, 150, 380], 0.99),
                (array(..., dtype=uint8), [150, 50, 299, 375], 0.98),
                (array(..., dtype=uint8), [120, 55, 300, 371], 0.96),
            ]
        """
        pass
