from contextlib import ExitStack
from unittest import TestCase
from unittest.mock import patch

from deepface.modules import streaming


class StreamWriterTests(TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name in ("__validate_opencv_gui", "build_demography_models",
                     "build_facial_recognition_model", "search_identity"):
            stack.enter_context(patch.object(streaming, name))
        self.cv = stack.enter_context(patch.object(streaming, "cv2"))
        self.cap = self.cv.VideoCapture.return_value
        self.cap.get.side_effect = [640, 480, 30]
        self.cap.read.return_value = (False, None)

    def test_filename_in_current_directory(self):
        streaming.analysis("gallery", output_path="recording.mp4")
        self.assertEqual(self.cv.VideoWriter.call_args.args[0], "recording.mp4")
        self.cap.release.assert_called_once()

    def test_failed_writer_releases_capture_before_first_frame(self):
        writer = self.cv.VideoWriter.return_value
        writer.isOpened.return_value = False
        with self.assertRaisesRegex(ValueError, "Cannot open video output"):
            streaming.analysis("gallery", output_path="./recording.mp4")
        self.cap.read.assert_not_called()
        writer.release.assert_called_once()
        self.cap.release.assert_called_once()
