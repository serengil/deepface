"""Optional MediaPipe installations must expose the detector's Solutions API."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from deepface.models.face_detection.external.MediaPipe import MediaPipeClient


def test_missing_dependency_recommends_a_compatible_install():
    with patch.dict("sys.modules", {"mediapipe": None}):
        with pytest.raises(ImportError, match="mediapipe>=0.8.7.3,<0.10.30"):
            MediaPipeClient()


def test_tasks_only_install_has_an_actionable_error():
    with patch.dict("sys.modules", {"mediapipe": SimpleNamespace(tasks=object())}):
        with pytest.raises(ImportError, match="mediapipe>=0.8.7.3,<0.10.30"):
            MediaPipeClient()


@pytest.mark.parametrize("selection", [0, 1])
def test_solutions_install_retains_both_detector_modes(monkeypatch, selection):
    monkeypatch.setenv("MEDIAPIPE_MODEL_SELECTION", str(selection))
    monkeypatch.setenv("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.6")
    constructor = Mock()
    module = SimpleNamespace(
        solutions=SimpleNamespace(face_detection=SimpleNamespace(FaceDetection=constructor))
    )
    with patch.dict("sys.modules", {"mediapipe": module}):
        client = MediaPipeClient()
    constructor.assert_called_once_with(min_detection_confidence=0.6, model_selection=selection)
    assert client.model is constructor.return_value
