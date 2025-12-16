
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

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
        norm_img = self.get_face_crop(img, (112, 112))
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

    def get_face_crop(self, img: np.ndarray, target_size: tuple[int, int], scale_face_area: float = 1.0, eyes_aligned: bool = False) -> np.ndarray:
        """
        Crops a face, aligns it using eyes positions (optional), ensures it has the size target_size.
            Pads with black pixels only if the crop goes outside the image boundaries.

        Args:
            img (np.ndarray): The source image.
            target_size (tuple[int, int]): Desired output size (width, height).
            scale_face_area (float): Scale factor (1.0 = tight face box, >1.0 = context).
            eyes_aligned (bool): Whether to rotate the image so eyes are horizontal.

        Returns:
            np.ndarray: The processed image (W=target_size[0], H=target_size[1]).
        """
        x, y, w, h = self.landmarks.x, self.landmarks.y, self.landmarks.w, self.landmarks.h
        
        # 1. Calculate the center of the face
        cx = x + w // 2
        cy = y + h // 2

        # 2. Alignment (Rotation)
        # We rotate the entire image around the face center (cx, cy).
        # This is safe because (cx, cy) remains the pivot point.
        if eyes_aligned and self.landmarks.left_eye and self.landmarks.right_eye:
            left_eye = self.landmarks.left_eye   # Tuple (x, y) - typically subject's left (image right)
            right_eye = self.landmarks.right_eye # Tuple (x, y) - typically subject's right (image left)

            # Calculate angle. 
            # Note: Ensure these point to the correct eye landmarks for your model
            dy = left_eye[1] - right_eye[1]
            dx = left_eye[0] - right_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # Rotate Matrix: Center at (cx, cy)
            M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
            
            # Apply rotation
            h_img, w_img = img.shape[:2]
            img = cv2.warpAffine(img, M, (w_img, h_img))

        # 3. Square the Bounding Box
        # To guarantee a square output without squashing, we use the largest dimension
        box_len = int(max(w, h) * scale_face_area)

        # 4. Calculate Crop Coordinates (Centered at cx, cy)
        half_len = box_len // 2
        x1 = cx - half_len
        y1 = cy - half_len
        x2 = x1 + box_len
        y2 = y1 + box_len

        # 5. Handle Boundary Conditions (Padding)
        h_img, w_img = img.shape[:2]
        
        # Calculate the intersection of the crop box and the image
        x1_valid = max(0, x1)
        y1_valid = max(0, y1)
        x2_valid = min(w_img, x2)
        y2_valid = min(h_img, y2)

        # Crop the valid part
        crop = img[y1_valid:y2_valid, x1_valid:x2_valid]

        # 6. Create the Canvas (Black Background)
        # We initialize a black square of size box_len x box_len
        padded_crop = np.zeros((box_len, box_len, 3), dtype=img.dtype)

        # Calculate where to paste the valid crop
        pad_x = x1_valid - x1
        pad_y = y1_valid - y1
        
        h_c, w_c = crop.shape[:2]
        
        # Paste
        padded_crop[pad_y : pad_y + h_c, pad_x : pad_x + w_c] = crop

        # 7. Final Resize
        # The padded_crop is now a square of size 'box_len'. 
        # We resize it to the requested 'target_size'.
        return cv2.resize(padded_crop, target_size)

    def to_dict(self):
        return {
            "confidence": self.confidence,
            "landmarks": self.landmarks.to_dict(),
            "quality": self.quality.to_dict() if self.quality else None,
            "pose": self.pose.to_dict() if self.pose else None,
            "demographics": self.demographics.to_dict() if self.demographics else None,
            "spoofing": self.spoofing.to_dict() if self.spoofing else None,
        }