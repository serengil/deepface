from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from deepface.models.face_detection.opencv import Ssd


class SsdCoordinateTests(TestCase):
    def test_fractional_model_pixels_survive_on_large_images(self):
        detector = object.__new__(Ssd.SsdClient)
        detector.model = Mock()
        detector.model.forward.return_value = np.array(
            [[[[0, 1, 0.99, 0.1259, 0.251, 0.1281, 0.497]]]], dtype=np.float32
        )
        with patch.object(Ssd, "cv2") as cv:
            cv.resize.return_value = np.zeros((300, 300, 3), dtype=np.uint8)
            faces = detector.detect_faces(np.zeros((900, 1200, 3), dtype=np.uint8))
        self.assertEqual(len(faces), 1)
        box = faces[0]
        self.assertEqual((box.x, box.y, box.w, box.h), (151, 225, 2, 222))
