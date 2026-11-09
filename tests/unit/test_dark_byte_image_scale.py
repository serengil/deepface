from unittest import TestCase
from unittest.mock import patch

import numpy as np

from deepface.modules import detection, preprocessing


class DarkImageScaleTests(TestCase):
    def test_extract_faces_distinguishes_bytes_from_normalized_floats(self):
        for dtype, expected in ((np.uint8, 1 / 255), (np.float32, 1.0)):
            with self.subTest(dtype=dtype):
                img = np.ones((2, 3, 3), dtype=dtype)
                face = detection.extract_faces(img, detector_backend="skip")[0]["face"]
                np.testing.assert_allclose(face, expected)

    def test_model_input_distinguishes_bytes_from_normalized_floats(self):
        for dtype, expected in ((np.uint8, 1 / 255), (np.float32, 1.0)):
            with self.subTest(dtype=dtype):
                img = np.ones((2, 3, 3), dtype=dtype)
                with patch.object(preprocessing.cv2, "resize", return_value=img, create=True):
                    batch = preprocessing.resize_image(img, (2, 3))
                self.assertEqual(batch.dtype, np.float32)
                np.testing.assert_allclose(batch, expected)
