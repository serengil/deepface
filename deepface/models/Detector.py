from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
from PIL import Image

# Notice that all facial detector models must be inherited from this class


class Detector(ABC):
    @abstractmethod
    def detect_faces(self, img: np.ndarray, align: bool = True) -> list:
        pass

    def align_face(
        self,
        img: np.ndarray,
        left_eye: Optional[Union[list, tuple]] = None,
        right_eye: Optional[Union[list, tuple]] = None,
    ) -> np.ndarray:
        """
        Align a given image horizantally with respect to their left and right eye locations
        Args:
            img (np.ndarray): pre-loaded image with detected face
            left_eye (list or tuple): coordinates of left eye with respect to the you
            right_eye(list or tuple): coordinates of right eye with respect to the you
        Returns:
            img (np.ndarray): aligned facial image
        """
        # if eye could not be detected for the given image, return image itself
        if left_eye is None or right_eye is None:
            return img

        # sometimes unexpectedly detected images come with nil dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            return img

        angle = float(
            np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        )
        img = Image.fromarray(img)
        img = np.array(img.rotate(angle))
        return img
