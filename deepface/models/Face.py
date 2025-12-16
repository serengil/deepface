
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

from deepface.models.spoofing.FasNet import crop

@dataclass
class FaceQuality:
    sharpness: float
    brightness: float
    contrast: float

    def to_dict(self):
        return {
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "contrast": self.contrast,
        }



@dataclass
class FaceSpoofing:
    spoof_confidence: float
    real_confidence: float
    uncertainty_confidence: float

    def to_dict(self):
        return {
            "spoof_confidence": self.spoof_confidence,
            "real_confidence": self.real_confidence,
            "uncertainty_confidence": self.uncertainty_confidence,
        }



@dataclass
class FaceLandmarks:
    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None
    nose: Optional[Tuple[int, int]] = None

    def is_within_boundaries(self, p: Optional[Tuple[int, int]], width: int, height: int) -> bool:
        return p is not None and 0 <= p[0] < width and 0 <= p[1] < height

    def is_valid(self, img_width: int, img_height: int) -> bool:
        """Check if all landmarks (nose is excluded as it is very much optional) are valid and within image boundaries.
        Args:
            img_width (int): Width of the image.
            img_height (int): Height of the image.
        Returns:
            bool: True if all landmarks are valid and within boundaries, False otherwise.
        """
        return (self.w > 0 and self.h > 0 and
                self.is_within_boundaries(self.left_eye, img_width, img_height) and
                self.is_within_boundaries(self.right_eye, img_width, img_height) and
                self.is_within_boundaries(self.mouth_left, img_width, img_height) and
                self.is_within_boundaries(self.mouth_right, img_width, img_height))

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "mouth_right": self.mouth_right,
            "mouth_left": self.mouth_left,
            "nose": self.nose,
        }



@dataclass
class FacePose:
    roll: float
    pitch: float
    yaw: float

    def to_dict(self):
        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }



@dataclass
class FaceDemographics:
    emotions: dict[str, float]
    genders: dict[str, float]
    races: dict[str, float]
    age: Optional[int] = None

    def dominant_emotion(self) -> Optional[str]:
        if self.emotions:
            return max(self.emotions, key=lambda k: self.emotions[k])
        return None

    def dominant_gender(self) -> Optional[str]:
        if self.genders:
            return max(self.genders, key=lambda k: self.genders[k])
        return None

    def dominant_race(self) -> Optional[str]:
        if self.races:
            return max(self.races, key=lambda k: self.races[k])
        return None

    def to_dict(self):
        return {
            "age": self.age,
            "emotions": self.emotions,
            "dominant_emotion": self.dominant_emotion(),
            "genders": self.genders,
            "dominant_gender": self.dominant_gender(),
            "races": self.races,
            "dominant_race": self.dominant_race(),
        }



@dataclass
class Face:
    """
    Initialize a Face object.
    Args:
        confidence (float): confidence score for face detection
        landmarks (FaceLandmarks): facial landmarks associated with the face
    """
    landmarks: FaceLandmarks
    confidence: Optional[float]
    quality: Optional[FaceQuality] = None
    pose: Optional[FacePose] = None
    demographics: Optional[FaceDemographics] = None
    spoofing: Optional[FaceSpoofing] = None
    embedding: Optional[List[float]] = None

    def __init__(self, img: np.ndarray, landmarks: FaceLandmarks, confidence: Optional[float]):
        """
        Initialize a Face object and automatically compute quality metrics and face 
        orientation (if possible) for later use.
        Args:
            img (np.ndarray): the image the face originates from
            confidence (float): confidence score from the face detector
            landmarks (FaceLandmarks): facial landmarks associated with the face
        """
        self.landmarks = landmarks
        self.confidence = confidence
        self.compute_quality(img)
        self.compute_pose()

    def compute_quality(self, img: np.ndarray):
        """
        Compute quality metrics for the face image and update the quality attribute.
        """
        # Note: we use the same crop function from the spoofing module to get
        # a natural quality and keep face ratio (no black borders)
        norm_img = crop(
            img, 
            (self.landmarks.x, self.landmarks.y, self.landmarks.w, self.landmarks.h),
            1, 
            112,
            112
        )
        if len(norm_img.shape) == 3:
            norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
        self.quality = FaceQuality(
            sharpness=cv2.Laplacian(norm_img, cv2.CV_64F).var(),
            brightness=float(norm_img.mean()),
            contrast=norm_img.std()
        )

    def compute_pose(self):
       # TODO: implement pose estimation.
       pass

    def get_face_crop(self, img: np.ndarray, target_size: tuple[int, int], aligned: bool = True) -> np.ndarray:
        """
        Generates a cropped and resized image of the face.
            The cropping is done based on the face landmarks if aligned is True, otherwise based on the face bounding box.

        Args:
            img (np.ndarray): The source image.
            target_size (tuple[int, int]): Desired output size (width, height).
            aligned (bool): Whether to use landmarks for alignment. Defaults to True.

        Returns:
            np.ndarray: The cropped and resized face image.
        """
        if aligned:
            # Src points from ArcFace (112x112) matrice used for alignment.
            src_points = np.array([
                [38.2946, 51.6963],  # Right Eye
                [73.5318, 51.5014],  # Left Eye
                [56.0252, 71.7366],  # Nose
                [41.5493, 92.3655],  # Right Mouth
                [70.7299, 92.2041]   # Left Mouth
            ], dtype=np.float32)

            # Apply scaling to the reference points if target size differs from 112x112.
            scale_x = target_size[0] / 112
            scale_y = target_size[1] / 112
            src_points[:, 0] *= scale_x
            src_points[:, 1] *= scale_y

            dst_points = np.array([
                self.landmarks.right_eye,
                self.landmarks.left_eye,
                self.landmarks.nose,
                self.landmarks.mouth_right,
                self.landmarks.mouth_left
            ], dtype=np.float32)

            tform_matrix, _ = cv2.estimateAffinePartial2D(dst_points, src_points, method=cv2.LMEDS)
            return cv2.warpAffine(img, tform_matrix, target_size, borderValue=0)
        else:
            src_points = np.array([
                [0, 0],
                [target_size[0] - 1, 0],
                [target_size[0] - 1, target_size[1] - 1],
                [0, target_size[1] - 1]
            ], dtype=np.float32)

            dst_points = np.array([
                [self.landmarks.x, self.landmarks.y],
                [self.landmarks.x + self.landmarks.w, self.landmarks.y],
                [self.landmarks.x + self.landmarks.w, self.landmarks.y + self.landmarks.h],
                [self.landmarks.x, self.landmarks.y + self.landmarks.h]
            ], dtype=np.float32)

            tform_matrix, _ = cv2.estimateAffinePartial2D(dst_points, src_points, method=cv2.LMEDS)
            return cv2.warpAffine(img, tform_matrix, target_size, borderValue=0)

    def to_dict(self):
        return {
            "confidence": self.confidence,
            "landmarks": self.landmarks.to_dict(),
            "quality": self.quality.to_dict() if self.quality else None,
            "pose": self.pose.to_dict() if self.pose else None,
            "demographics": self.demographics.to_dict() if self.demographics else None,
            "spoofing": self.spoofing.to_dict() if self.spoofing else None,
        }