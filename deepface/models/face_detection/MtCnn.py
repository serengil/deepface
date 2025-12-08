# built-in dependencies
from typing import List, Any

# 3rd party dependencies
from numpy.typing import NDArray
from mtcnn import MTCNN

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion


# pylint: disable=too-few-public-methods
class MtCnnClient(Detector):
    """
    Class to cover common face detection functionalitiy for MtCnn backend
    """

    def __init__(self) -> None:
        self.model = MTCNN()

    def detect_faces(self, img: NDArray[Any]) -> List[FacialAreaRegion]:
        """
        Detect and align face with mtcnn

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """

        resp = []

        # mtcnn expects RGB but OpenCV read BGR
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img[:, :, ::-1]
        detections = self.model.detect_faces(img_rgb)

        if detections is not None and len(detections) > 0:

            for current_detection in detections:
                x, y, w, h = current_detection["box"]
                confidence = current_detection["confidence"]
                # mtcnn detector assigns left eye with respect to the observer
                # but we are setting it with respect to the person itself
                left_eye = current_detection["keypoints"]["right_eye"]
                right_eye = current_detection["keypoints"]["left_eye"]

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
