# built-in dependencies
import logging
from typing import List, Union

# 3rd party dependencies
import numpy as np
from mtcnn import MTCNN

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion

logger = logging.getLogger(__name__)

# pylint: disable=too-few-public-methods
class MtCnnClient(Detector):
    """
    Class to cover common face detection functionalitiy for MtCnn backend
    """

    def __init__(self):
        self.model = MTCNN()
        self.supports_batch_detection = self._supports_batch_detection()

    def detect_faces(
        self,
        img: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]:
        """
        Detect and align faces with mtcnn for a list of images

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): 
                pre-loaded image as numpy array or a list of those

        Returns:
            results (Union[List[FacialAreaRegion], List[List[FacialAreaRegion]]]): 
                A list of FacialAreaRegion objects for a single image 
                or a list of lists of FacialAreaRegion objects for each image
        """

        if not isinstance(img, list):
            img = [img]

        resp = []

        # mtcnn expects RGB but OpenCV read BGR
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = [img[:, :, ::-1] for img in img]
        if self.supports_batch_detection:
            detections = self.model.detect_faces(img_rgb)
        else:
            detections = [self.model.detect_faces(single_img) for single_img in img_rgb]

        for image_detections in detections:
            image_resp = []
            if image_detections is not None and len(image_detections) > 0:
                for current_detection in image_detections:
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

                    image_resp.append(facial_area)

            resp.append(image_resp)

        if len(resp) == 1:
            return resp[0]
        return resp
    
    def _supports_batch_detection(self) -> bool:
        import mtcnn
        try:
            mtcnn_version = mtcnn.__version__
        except AttributeError:
            try:
                import mtcnn.metadata
                mtcnn_version = mtcnn.metadata.__version__
            except AttributeError:
                logger.warning("Failed to determine mtcnn version")
                logger.warning("Fallback to single image detection")
                return False
        supports_batch_detection = mtcnn_version >= "1.0.0"
        if not supports_batch_detection:
            logger.warning("MtCnn version is less than 1.0.0, batch detection is not supported")
            logger.warning("Fallback to single image detection")
        return supports_batch_detection
