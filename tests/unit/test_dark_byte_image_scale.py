from unittest import TestCase

import numpy as np

from deepface.modules import detection, preprocessing


class DarkImageScaleTests(TestCase):
    def test_extract_faces_distinguishes_bytes_from_normalized_floats(self):
        for dtype, expected in (
            (np.uint8, 1 / 255),
            (np.float32, 1.0),
            (np.float64, 1.0),
        ):
            with self.subTest(dtype=dtype):
                img = np.ones((2, 3, 3), dtype=dtype)
                face = detection.extract_faces(img, detector_backend="skip")[0]["face"]
                np.testing.assert_allclose(face, expected)

    def test_model_input_distinguishes_bytes_from_normalized_floats(self):
        for dtype, expected in (
            (np.uint8, 1 / 255),
            (np.float32, 1.0),
            (np.float64, 1.0),
        ):
            with self.subTest(dtype=dtype):
                img = np.ones((2, 3, 3), dtype=dtype)
                batch = preprocessing.resize_image(img, (2, 3))
                self.assertEqual(batch.shape, (1, 2, 3, 3))
                np.testing.assert_allclose(batch, expected)
                self.assertEqual(batch.dtype, np.float32)
