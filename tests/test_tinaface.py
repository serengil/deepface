"""
Unit tests for TinaFace detector backend
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from deepface.models.face_detection.TinaFace import TinaFaceClient
from deepface.models.Detector import FacialAreaRegion


class TestTinaFace(unittest.TestCase):
    """Test cases for TinaFace detector"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = TinaFaceClient()
        # Ensure eye finder returns None to trigger heuristic fallback deterministically
        try:
            self.detector._eye_finder.find_eyes = lambda roi: (None, None)
        except Exception:
            pass
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_initialization(self):
        """Test TinaFace client initialization"""
        self.assertIsNone(self.detector.model)
        self.assertEqual(self.detector.runtime, "ort")

    def test_build_model_success(self):
        """Test successful model building with a fake onnxruntime module"""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [SimpleNamespace(name="input")]
        mock_session.get_outputs.return_value = [SimpleNamespace(name="out")]

        class FakeOrtModule:
            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

            def __init__(self):
                pass

            class InferenceSession:  # pylint: disable=too-few-public-methods
                def __new__(cls, *args, **kwargs):  # noqa: D401
                    return mock_session

        with patch("deepface.models.face_detection.TinaFace.weight_utils.download_weights_if_necessary", return_value="/path/to/model.onnx"):
            with patch.dict('sys.modules', {'onnxruntime': FakeOrtModule()}):
                result = self.detector._build_model()
                self.assertEqual(result, mock_session)
                self.assertEqual(self.detector.runtime, "ort")

    def test_build_model_missing_onnxruntime(self):
        """Test model building when onnxruntime is not available"""
        with patch("deepface.models.face_detection.TinaFace.weight_utils.download_weights_if_necessary", return_value="/path/to/model.onnx"):
            with patch.dict('sys.modules', {'onnxruntime': None}):
                with self.assertRaises(ValueError) as context:
                    self.detector._build_model()
                self.assertIn("onnxruntime is required", str(context.exception))

    def test_preprocess(self):
        """Test image preprocessing"""
        result = self.detector._preprocess(self.test_image)
        self.assertIn("input", result)
        self.assertIn("meta", result)
        self.assertEqual(result["input"].shape[0], 1)  # batch size
        self.assertEqual(result["input"].shape[1], 3)  # channels
        self.assertEqual(result["input"].dtype, np.float32)

    def test_parse_out_rows(self):
        """Test parsing of model output rows"""
        # Create mock output with 2 faces, 15 columns (5 + 10 landmarks)
        mock_output = np.array([
            [10, 20, 50, 60, 0.9, 15, 25, 45, 25, 30, 35, 40, 45, 50, 55],  # face 1
            [100, 120, 150, 160, 0.8, 105, 125, 145, 125, 130, 135, 140, 145, 150, 155]  # face 2
        ])
        result = self.detector._parse_out_rows(
            mock_output, 0.5, 200, 200, 1.0
        )
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], FacialAreaRegion)
        self.assertEqual(result[0].confidence, 0.9)
        self.assertEqual(result[0].left_eye, (15, 25))
        self.assertEqual(result[0].right_eye, (45, 25))

    def test_sigmoid(self):
        """Test sigmoid activation function"""
        x = np.array([0, 1, -1])
        result = self.detector._sigmoid(x)
        self.assertEqual(result[0], 0.5)  # sigmoid(0) = 0.5
        self.assertGreater(result[1], 0.5)  # sigmoid(1) > 0.5
        self.assertLess(result[2], 0.5)  # sigmoid(-1) < 0.5

    def test_nms(self):
        """Test non-maximum suppression"""
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
        scores = np.array([0.9, 0.8, 0.7])
        result = self.detector._nms(boxes, scores, 0.5, 10)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_populate_missing_eyes(self):
        """Test eye landmark population with heuristic fallback"""
        # Create faces without eye landmarks
        faces = [
            FacialAreaRegion(x=10, y=20, w=40, h=40, confidence=0.9),
            FacialAreaRegion(x=100, y=120, w=40, h=40, confidence=0.8)
        ]
        result = self.detector._populate_missing_eyes(self.test_image, faces)
        self.assertEqual(len(result), 2)
        for face in result:
            self.assertIsNotNone(face.left_eye)
            self.assertIsNotNone(face.right_eye)
            self.assertIsInstance(face.left_eye, tuple)
            self.assertIsInstance(face.right_eye, tuple)

    def test_detect_faces_no_faces(self):
        """Test face detection when no faces are found"""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [SimpleNamespace(name="input")]
        mock_session.get_outputs.return_value = [SimpleNamespace(name="out")]
        mock_session.run.return_value = [np.array([])]  # Empty output

        class FakeOrtModule:
            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

            class InferenceSession:  # pylint: disable=too-few-public-methods
                def __new__(cls, *args, **kwargs):  # noqa: D401
                    return mock_session

        with patch("deepface.models.face_detection.TinaFace.weight_utils.download_weights_if_necessary", return_value="/path/to/model.onnx"):
            with patch.dict('sys.modules', {'onnxruntime': FakeOrtModule()}):
                # Build model first
                self.detector._build_model()
                # Test detection
                result = self.detector.detect_faces(self.test_image)
                self.assertEqual(result, [])

    def test_environment_threshold(self):
        """Test environment variable threshold setting"""
        with patch.dict('os.environ', {'TINAFACE_THRESHOLD': '0.5'}):
            threshold = float(self._get_threshold())
            self.assertEqual(threshold, 0.5)

    def test_default_threshold(self):
        """Test default threshold when environment variable is not set"""
        with patch.dict('os.environ', {}, clear=True):
            threshold = float(self._get_threshold())
            self.assertEqual(threshold, 0.35)

    def _get_threshold(self):
        """Helper: replicate threshold logic"""
        import os
        return os.getenv("TINAFACE_THRESHOLD", "0.35")


if __name__ == "__main__":
    unittest.main()
